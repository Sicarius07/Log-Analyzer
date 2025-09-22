import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Search, AlertCircle, CheckCircle, DollarSign, FileText, Zap, ChevronDown, ChevronRight, Server, Clock, AlertTriangle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './components/ui/collapsible';
import './App.css';

interface LogEntry {
  [key: string]: any;
}

interface AnalysisResult {
  relevant_logs: LogEntry[];
  analysis: string;
  cost_info: {
    total_tokens: number;
    total_cost: number;
    input_tokens: number;
    output_tokens: number;
  };
  filtered_count: number;
  total_count: number;
}

interface IndexingProgress {
  status: string;
  progress: number;
  indexed_count: number;
  total_count: number;
  message: string;
  session_id: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [indexingProgress, setIndexingProgress] = useState<IndexingProgress | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);

  // Set dark theme on mount
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      // Start pre-indexing immediately
      startIndexing(selectedFile);
    }
  };

  const startIndexing = async (fileToIndex: File) => {
    setIsIndexing(true);
    setIndexingProgress(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', fileToIndex);

    try {
      const response = await axios.post('http://localhost:8000/index-logs', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const initialProgress: IndexingProgress = response.data;
      setIndexingProgress(initialProgress);

      // Poll for progress updates
      pollIndexingProgress(initialProgress.session_id);
    } catch (err: any) {
      console.error('Indexing error:', err);
      setError(err.response?.data?.detail || 'Error starting indexing');
      setIsIndexing(false);
    }
  };

  const pollIndexingProgress = async (sessionId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:8000/index-progress/${sessionId}`);
        const progress: IndexingProgress = response.data;
        
        setIndexingProgress(progress);

        if (progress.status === 'completed' || progress.status === 'error') {
          clearInterval(pollInterval);
          setIsIndexing(false);
          
          if (progress.status === 'error') {
            setError(progress.message);
          }
        }
      } catch (err) {
        console.error('Error polling progress:', err);
        clearInterval(pollInterval);
        setIsIndexing(false);
      }
    }, 1000); // Poll every second
  };

  const handleAnalyze = async () => {
    if (!file || !prompt.trim()) return;

    setLoading(true);
    setError(null);

    // Get fresh file reference from the input element to avoid ERR_UPLOAD_FILE_CHANGED
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const currentFile = fileInput?.files?.[0];
    
    if (!currentFile) {
      setError('Please select a file');
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('prompt', prompt);
    
    // Include session_id if we have indexing progress
    if (indexingProgress?.session_id) {
      formData.append('session_id', indexingProgress.session_id);
    }

    try {
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log('Backend response:', response.data); // Debug log
      
      // Ensure cost_info has all required fields
      const resultData = {
        ...response.data,
        cost_info: {
          total_tokens: 0,
          input_tokens: 0,
          output_tokens: 0,
          total_cost: 0,
          ...response.data.cost_info
        }
      };
      
      setResult(resultData);
    } catch (err: any) {
      console.error('Analysis error:', err); // Debug log
      setError(err.response?.data?.detail || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  const LogEntryComponent = ({ log, index }: { log: LogEntry; index: number }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    
    // Extract metadata for highlights
    const service = log.resource_attributes?.['service.name'] || 'Unknown Service';
    const timestamp = log.timestamp ? 
      new Date(parseInt(log.timestamp.toString()) / 1000000).toLocaleString() : 
      'Unknown time';
    const body = log.body || 'No message';
    const severity = log.fields?.severity_text || 'INFO';
    const severityNumber = log.fields?.severity_number;
    const traceId = log.fields?.trace_id;
    const spanId = log.fields?.span_id;
    const deployment = log.resource_attributes?.['k8s.deployment.name'];
    const namespace = log.resource_attributes?.['k8s.namespace.name'];
    const podName = log.resource_attributes?.['k8s.pod.name'];

    const getSeverityVariant = (severity: string) => {
      switch (severity.toLowerCase()) {
        case 'error':
        case 'fatal':
          return 'destructive';
        case 'warn':
          return 'outline';
        default:
          return 'secondary';
      }
    };

    return (
      <Card className="log-entry">
        <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
          <CollapsibleTrigger asChild>
            <div className="log-header">
              <div className="log-metadata">
                <Badge variant="default" className="service-name">
                  <Server className="w-3 h-3 mr-1" />
                  {service}
                </Badge>
                <Badge variant={getSeverityVariant(severity)} className="severity">
                  <AlertTriangle className="w-3 h-3 mr-1" />
                  {severity}
                  {severityNumber && <span className="severity-number ml-1">({severityNumber})</span>}
                </Badge>
                {traceId && (
                  <Badge variant="outline" className="trace-id" title={`Trace ID: ${traceId}`}>
                    🔗 {traceId.substring(0, 8)}...
                  </Badge>
                )}
                {spanId && (
                  <Badge variant="outline" className="span-id" title={`Span ID: ${spanId}`}>
                    📍 {spanId.substring(0, 8)}...
                  </Badge>
                )}
                {deployment && (
                  <Badge variant="outline" className="deployment" title={`Deployment: ${deployment}`}>
                    🚀 {deployment}
                  </Badge>
                )}
                {namespace && (
                  <Badge variant="outline" className="namespace" title={`Namespace: ${namespace}`}>
                    🏠 {namespace}
                  </Badge>
                )}
              </div>
              <div className="log-controls">
                <span className="timestamp flex items-center">
                  <Clock className="w-3 h-3 mr-1" />
                  {timestamp}
                </span>
                <span className="expand-icon">
                  {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </span>
              </div>
            </div>
          </CollapsibleTrigger>
          
          {!isExpanded && (
            <div className="log-preview">
              <div className="log-body-preview">{body.substring(0, 200)}{body.length > 200 ? '...' : ''}</div>
            </div>
          )}
          
          <CollapsibleContent>
            <div className="log-expanded">
              <div className="log-body-full">
                <strong>Message:</strong> {body}
              </div>
              {podName && (
                <div className="log-detail">
                  <strong>Pod:</strong> {podName}
                </div>
              )}
              <div className="log-json">
                <strong>Complete Log (JSON):</strong>
                <pre className="json-content">
                  {JSON.stringify(log, null, 2)}
                </pre>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </Card>
    );
  };

  return (
    <div className="App dark">
      <div className="container">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-primary text-primary-foreground rounded-lg">
              <Zap className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Relvy Log Analyzer</h1>
              <p className="text-muted-foreground">AI That Debugs With You</p>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold mb-2">Debug Faster with AI-Powered Log Analysis</h2>
          <p className="text-muted-foreground">Upload your logs and describe the incident. Our AI will filter and analyze relevant entries to help you find the root cause.</p>
        </section>

        {/* Upload and Analysis Section */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Upload Log File
            </CardTitle>
            <CardDescription>
              Upload your NDJSON log file for AI-powered analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="file-upload">
              <input
                type="file"
                id="file-input"
                accept=".ndjson,.jsonl"
                onChange={handleFileChange}
                className="hidden"
              />
              <Button asChild variant="outline" className="w-full">
                <label htmlFor="file-input" className="cursor-pointer">
                  <Upload className="w-4 h-4 mr-2" />
                  {file ? file.name : 'Choose NDJSON file'}
                </label>
              </Button>
            </div>

            <div className="space-y-2">
              <label htmlFor="prompt" className="text-sm font-medium">
                Describe the incident or issue:
              </label>
              <textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="e.g., cart service is crashing, check logs for errors"
                className="prompt-input"
                rows={3}
              />
            </div>

            <Button
              onClick={handleAnalyze}
              disabled={!file || !prompt.trim() || loading || isIndexing}
              className="w-full"
              size="lg"
            >
              {loading ? (
                <>
                  <div className="loading-spinner mr-2"></div>
                  {indexingProgress?.session_id ? 'Analyzing (Pre-indexed)...' : 'Analyzing...'}
                </>
              ) : isIndexing ? (
                <>
                  <div className="loading-spinner mr-2"></div>
                  Please wait - Indexing in progress...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  {indexingProgress?.status === 'completed' ? 'Analyze Logs (Fast)' : 'Analyze Logs'}
                </>
              )}
            </Button>
          </CardContent>
          </Card>

        {/* Indexing Progress */}
        {(isIndexing || indexingProgress) && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Indexing Logs
              </CardTitle>
              <CardDescription>
                Pre-processing logs for faster analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {indexingProgress && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <span>{indexingProgress.message}</span>
                    <span>{Math.round(indexingProgress.progress * 100)}%</span>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div 
                      className="bg-primary h-2 rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${indexingProgress.progress * 100}%` }}
                    ></div>
                  </div>
                  
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{indexingProgress.indexed_count} indexed</span>
                    <span>{indexingProgress.total_count} total logs</span>
                  </div>
                  
                  {indexingProgress.status === 'completed' && (
                    <div className="flex items-center gap-2 text-green-600 text-sm">
                      <CheckCircle className="w-4 h-4" />
                      Indexing completed! Ready for analysis.
                    </div>
                  )}
                </div>
              )}
              
              {isIndexing && !indexingProgress && (
                <div className="flex items-center gap-2">
                  <div className="loading-spinner"></div>
                  <span>Starting indexing...</span>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Error Display */}
        {error && (
          <Card className="mb-8 border-destructive">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-5 h-5" />
                <span className="font-medium">Error</span>
              </div>
              <p className="mt-2 text-sm">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Cost Information */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="w-5 h-5" />
                  Analysis Cost
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Total Tokens:</span>
                    <div className="font-medium">{(result.cost_info?.total_tokens || 0).toLocaleString()}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Input Tokens:</span>
                    <div className="font-medium">{(result.cost_info?.input_tokens || 0).toLocaleString()}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Output Tokens:</span>
                    <div className="font-medium">{(result.cost_info?.output_tokens || 0).toLocaleString()}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Total Cost:</span>
                    <div className="font-medium">${(result.cost_info?.total_cost || 0).toFixed(4)}</div>
                  </div>
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  Filtered {result.filtered_count} relevant logs from {result.total_count} total logs
                </div>
              </CardContent>
            </Card>

            {/* AI Analysis */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  AI Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="analysis-content">
                  <ReactMarkdown>{result.analysis}</ReactMarkdown>
                </div>
              </CardContent>
            </Card>

            {/* Relevant Logs */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  Relevant Logs ({result.relevant_logs.length})
                </CardTitle>
                <CardDescription>
                  Logs identified as relevant to your incident
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {result.relevant_logs.map((log, index) => (
                    <LogEntryComponent key={index} log={log} index={index} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-16 py-8 border-t border-border">
          <div className="text-center text-sm text-muted-foreground">
            &copy; 2025 Relvy Log Analyzer. Built for debugging excellence.
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;