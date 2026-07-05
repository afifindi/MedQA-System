import React, { useEffect, useState } from 'react';
import { chatService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { CheckCircle2, XCircle, Server, Cpu, Layers } from 'lucide-react';

const AboutPage = () => {
  const [statusData, setStatusData] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [health, model] = await Promise.all([
          chatService.getHealth(),
          chatService.getModelInfo()
        ]);
        setStatusData(health);
        setModelData(model);
      } catch (err) {
        setError("Failed to connect to backend services.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const StatusIcon = ({ status }) => {
    return status === 'loaded' || status === 'healthy' ? (
      <CheckCircle2 className="w-5 h-5 text-green-500" />
    ) : (
      <XCircle className="w-5 h-5 text-red-500" />
    );
  };

  return (
    <div className="flex-1 overflow-y-auto bg-background p-6 lg:p-12">
      <div className="max-w-4xl mx-auto space-y-8">
        
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">About & System Status</h1>
          <p className="text-muted-foreground">
            This system uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate medical answers.
          </p>
        </div>

        {loading ? (
          <div className="py-20 flex justify-center"><LoadingSpinner size="lg" /></div>
        ) : error ? (
          <div className="p-4 bg-destructive/10 text-destructive border border-destructive/20 rounded-lg">
            {error}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* System Health */}
            <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <Server className="w-5 h-5 text-primary" />
                Backend Components
              </h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg border border-border/50">
                  <span className="font-medium text-sm">Overall Status</span>
                  <span className={`text-sm font-semibold uppercase ${statusData.status === 'healthy' ? 'text-green-500' : 'text-red-500'}`}>
                    {statusData.status}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Embedding Service</span>
                    <StatusIcon status={statusData.components?.embedding_service} />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">FAISS Retrieval</span>
                    <StatusIcon status={statusData.components?.retrieval_service} />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Knowledge Base</span>
                    <StatusIcon status={statusData.components?.knowledge_base_service} />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Gemma LLM</span>
                    <StatusIcon status={statusData.components?.gemma_service} />
                  </div>
                </div>
              </div>
            </div>

            {/* Model Configuration */}
            <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <Cpu className="w-5 h-5 text-primary" />
                AI Configuration
              </h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg border border-border/50">
                  <span className="font-medium text-sm">Compute Device</span>
                  <span className="text-sm font-semibold uppercase text-primary">
                    {modelData.device}
                  </span>
                </div>
                
                <div className="space-y-3">
                  <div>
                    <span className="text-xs text-muted-foreground uppercase tracking-wider block mb-1">Generator Model</span>
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Layers className="w-4 h-4 text-muted-foreground" />
                      {modelData.generator_model}
                    </div>
                  </div>
                  
                  <div>
                    <span className="text-xs text-muted-foreground uppercase tracking-wider block mb-1">Embedding Model</span>
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Database className="w-4 h-4 text-muted-foreground" />
                      {modelData.embedding_model}
                    </div>
                  </div>

                  <div>
                    <span className="text-xs text-muted-foreground uppercase tracking-wider block mb-1">LoRA & Config</span>
                    <div className="text-sm font-mono text-muted-foreground bg-muted p-2 rounded truncate mt-1">
                      {modelData.lora_path}
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
};

export default AboutPage;
