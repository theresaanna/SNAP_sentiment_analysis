import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Save, RotateCcw, Download, Upload, BarChart3 } from 'lucide-react';

const ManualTaggingTool = () => {
  const [replies, setReplies] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [manualTags, setManualTags] = useState({});
  const [showStats, setShowStats] = useState(false);
  const [filterMode, setFilterMode] = useState('all'); // 'all', 'untagged', 'conflicted'
  const [filteredIndices, setFilteredIndices] = useState([]);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load CSV data
  useEffect(() => {
    const loadData = async () => {
      try {
        const csvContent = await window.fs.readFile('threads_replies_18087086674818268_20250825_191726_analyzed.csv', { encoding: 'utf8' });

        // Simple CSV parser
        const lines = csvContent.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.replace(/"/g, ''));
        const data = lines.slice(1).map(line => {
          const values = [];
          let current = '';
          let inQuotes = false;

          for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
              inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
              values.push(current.replace(/"/g, ''));
              current = '';
            } else {
              current += char;
            }
          }
          values.push(current.replace(/"/g, ''));

          const obj = {};
          headers.forEach((header, index) => {
            obj[header] = values[index] || '';
          });
          return obj;
        });

        setReplies(data);
        setIsLoaded(true);

        // Load existing manual tags from localStorage if available
        const savedTags = localStorage.getItem('manual_sentiment_tags');
        if (savedTags) {
          setManualTags(JSON.parse(savedTags));
        }

      } catch (error) {
        console.error('Error loading data:', error);
      }
    };

    loadData();
  }, []);

  // Update filtered indices based on filter mode
  useEffect(() => {
    if (!replies.length) return;

    let indices = [];
    replies.forEach((reply, index) => {
      const hasManualTag = manualTags[reply.id];
      const isConflicted = hasConflictingSentiment(reply);

      switch (filterMode) {
        case 'untagged':
          if (!hasManualTag) indices.push(index);
          break;
        case 'conflicted':
          if (isConflicted && !hasManualTag) indices.push(index);
          break;
        default:
          indices.push(index);
      }
    });

    setFilteredIndices(indices);
    if (indices.length > 0 && !indices.includes(currentIndex)) {
      setCurrentIndex(indices[0]);
    }
  }, [filterMode, replies, manualTags, currentIndex]);

  const hasConflictingSentiment = (reply) => {
    if (!reply.vader_sentiment || !reply.textblob_sentiment || !reply.final_sentiment) return false;

    const sentiments = [reply.vader_sentiment, reply.textblob_sentiment, reply.final_sentiment];
    const uniqueSentiments = [...new Set(sentiments)];
    return uniqueSentiments.length > 1;
  };

  const handleManualTag = (sentiment) => {
    const currentReply = replies[currentIndex];
    if (!currentReply) return;

    const newTags = {
      ...manualTags,
      [currentReply.id]: sentiment
    };

    setManualTags(newTags);

    // Save to localStorage
    localStorage.setItem('manual_sentiment_tags', JSON.stringify(newTags));

    // Move to next untagged reply
    moveToNext();
  };

  const moveToNext = () => {
    const nextIndex = filteredIndices.find(i => i > currentIndex);
    if (nextIndex !== undefined) {
      setCurrentIndex(nextIndex);
    }
  };

  const moveToPrevious = () => {
    const prevIndex = [...filteredIndices].reverse().find(i => i < currentIndex);
    if (prevIndex !== undefined) {
      setCurrentIndex(prevIndex);
    }
  };

  const exportTaggedData = () => {
    const taggedData = replies.map(reply => ({
      ...reply,
      manual_sentiment: manualTags[reply.id] || '',
      is_manually_tagged: !!manualTags[reply.id],
      has_conflict: hasConflictingSentiment(reply)
    }));

    const csv = [
      Object.keys(taggedData[0]).join(','),
      ...taggedData.map(row =>
        Object.values(row).map(val =>
          typeof val === 'string' && val.includes(',') ? `"${val}"` : val
        ).join(',')
      )
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'manually_tagged_replies.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const getTaggingStats = () => {
    const totalReplies = replies.length;
    const taggedCount = Object.keys(manualTags).length;
    const conflictedCount = replies.filter(hasConflictingSentiment).length;
    const taggedConflictedCount = replies.filter(reply =>
      hasConflictingSentiment(reply) && manualTags[reply.id]
    ).length;

    const sentimentBreakdown = Object.values(manualTags).reduce((acc, sentiment) => {
      acc[sentiment] = (acc[sentiment] || 0) + 1;
      return acc;
    }, {});

    return {
      totalReplies,
      taggedCount,
      taggedPercent: ((taggedCount / totalReplies) * 100).toFixed(1),
      conflictedCount,
      taggedConflictedCount,
      conflictedTaggedPercent: conflictedCount > 0 ? ((taggedConflictedCount / conflictedCount) * 100).toFixed(1) : 0,
      sentimentBreakdown
    };
  };

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-lg">Loading replies data...</div>
      </div>
    );
  }

  if (!replies.length) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-lg">No data loaded. Please make sure the CSV file is available.</div>
      </div>
    );
  }

  const currentReply = replies[currentIndex];
  const currentFilteredIndex = filteredIndices.indexOf(currentIndex);
  const stats = getTaggingStats();

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Manual Sentiment Tagging</h1>
          <div className="flex gap-2">
            <button
              onClick={() => setShowStats(!showStats)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              <BarChart3 size={16} />
              Stats
            </button>
            <button
              onClick={exportTaggedData}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              <Download size={16} />
              Export
            </button>
          </div>
        </div>

        {showStats && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-bold mb-3">Tagging Progress</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="font-semibold">Total Replies</div>
                <div>{stats.totalReplies}</div>
              </div>
              <div>
                <div className="font-semibold">Tagged</div>
                <div>{stats.taggedCount} ({stats.taggedPercent}%)</div>
              </div>
              <div>
                <div className="font-semibold">Conflicted</div>
                <div>{stats.conflictedCount}</div>
              </div>
              <div>
                <div className="font-semibold">Conflicted Tagged</div>
                <div>{stats.taggedConflictedCount} ({stats.conflictedTaggedPercent}%)</div>
              </div>
            </div>
            <div className="mt-3">
              <div className="font-semibold mb-2">Manual Tags Breakdown:</div>
              <div className="flex gap-4 text-sm">
                {Object.entries(stats.sentimentBreakdown).map(([sentiment, count]) => (
                  <span key={sentiment} className="px-2 py-1 bg-white rounded">
                    {sentiment}: {count}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="flex gap-4 mb-4">
          <select
            value={filterMode}
            onChange={(e) => setFilterMode(e.target.value)}
            className="px-3 py-2 border rounded"
          >
            <option value="all">All Replies ({replies.length})</option>
            <option value="untagged">Untagged ({replies.length - stats.taggedCount})</option>
            <option value="conflicted">Conflicted Models ({stats.conflictedCount - stats.taggedConflictedCount})</option>
          </select>
        </div>

        <div className="flex items-center justify-between mb-4">
          <button
            onClick={moveToPrevious}
            disabled={currentFilteredIndex <= 0}
            className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded disabled:bg-gray-300 hover:bg-gray-600"
          >
            <ChevronLeft size={16} />
            Previous
          </button>

          <span className="text-sm text-gray-600">
            {currentFilteredIndex + 1} of {filteredIndices.length} ({filterMode})
          </span>

          <button
            onClick={moveToNext}
            disabled={currentFilteredIndex >= filteredIndices.length - 1}
            className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded disabled:bg-gray-300 hover:bg-gray-600"
          >
            Next
            <ChevronRight size={16} />
          </button>
        </div>

        {currentReply && (
          <div className="space-y-4">
            <div className="border rounded-lg p-4">
              <div className="flex justify-between items-start mb-3">
                <div className="font-semibold">@{currentReply.username}</div>
                <div className="text-sm text-gray-500">ID: {currentReply.id}</div>
              </div>

              <div className="text-lg mb-4 p-3 bg-gray-50 rounded">
                {currentReply.text || currentReply.cleaned_text || 'No text content'}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4 text-sm">
                <div>
                  <div className="font-semibold">VADER</div>
                  <div className={`px-2 py-1 rounded ${
                    currentReply.vader_sentiment === 'positive' ? 'bg-green-100' :
                    currentReply.vader_sentiment === 'negative' ? 'bg-red-100' : 'bg-gray-100'
                  }`}>
                    {currentReply.vader_sentiment} ({parseFloat(currentReply.vader_compound || 0).toFixed(3)})
                  </div>
                </div>

                <div>
                  <div className="font-semibold">TextBlob</div>
                  <div className={`px-2 py-1 rounded ${
                    currentReply.textblob_sentiment === 'positive' ? 'bg-green-100' :
                    currentReply.textblob_sentiment === 'negative' ? 'bg-red-100' : 'bg-gray-100'
                  }`}>
                    {currentReply.textblob_sentiment} ({parseFloat(currentReply.textblob_polarity || 0).toFixed(3)})
                  </div>
                </div>

                <div>
                  <div className="font-semibold">Combined</div>
                  <div className={`px-2 py-1 rounded ${
                    currentReply.final_sentiment === 'positive' ? 'bg-green-100' :
                    currentReply.final_sentiment === 'negative' ? 'bg-red-100' : 'bg-gray-100'
                  }`}>
                    {currentReply.final_sentiment} ({parseFloat(currentReply.combined_sentiment_score || 0).toFixed(3)})
                  </div>
                </div>
              </div>

              {manualTags[currentReply.id] && (
                <div className="mb-4 p-3 bg-blue-50 rounded">
                  <div className="font-semibold">Current Manual Tag:
                    <span className={`ml-2 px-2 py-1 rounded text-sm ${
                      manualTags[currentReply.id] === 'positive' ? 'bg-green-200' :
                      manualTags[currentReply.id] === 'negative' ? 'bg-red-200' : 'bg-gray-200'
                    }`}>
                      {manualTags[currentReply.id]}
                    </span>
                  </div>
                </div>
              )}

              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => handleManualTag('positive')}
                  className="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 font-semibold"
                >
                  👍 Positive
                </button>
                <button
                  onClick={() => handleManualTag('neutral')}
                  className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 font-semibold"
                >
                  😐 Neutral
                </button>
                <button
                  onClick={() => handleManualTag('negative')}
                  className="px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 font-semibold"
                >
                  👎 Negative
                </button>
              </div>

              <div className="mt-4 text-xs text-gray-500 grid grid-cols-2 gap-2">
                <div>Word Count: {currentReply.word_count}</div>
                <div>Emoji Count: {currentReply.emoji_count}</div>
                <div>Has Question: {currentReply.has_question}</div>
                <div>Has Exclamation: {currentReply.has_exclamation}</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Next Steps for Model Training</h2>
        <div className="space-y-3 text-sm">
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold">1</div>
            <div>
              <strong>Manual Tagging:</strong> Tag at least 100-200 replies for good training data. Focus on conflicted cases first.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold">2</div>
            <div>
              <strong>Export Data:</strong> Use the Export button to download your manually tagged dataset.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold">3</div>
            <div>
              <strong>Train Models:</strong> Use scikit-learn to train custom models on your tagged data (Naive Bayes, SVM, or Neural Networks).
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold">4</div>
            <div>
              <strong>Validate:</strong> Use cross-validation to measure improvement over the existing VADER/TextBlob approach.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManualTaggingTool;