import React from 'react';
import { Link } from 'react-router-dom';
import { Stethoscope, ArrowRight, ShieldCheck, Zap, Database } from 'lucide-react';

const LandingPage = () => {
  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16 sm:py-24">
        
        {/* Hero Section */}
        <div className="text-center space-y-8">
          <div className="inline-flex items-center justify-center p-4 bg-primary/10 rounded-2xl mb-4">
            <Stethoscope className="w-16 h-16 text-primary" />
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-foreground">
            Medical QA System
          </h1>
          <p className="max-w-2xl mx-auto text-xl text-muted-foreground">
            Get accurate, evidence-based answers to medical questions powered by advanced AI and Retrieval-Augmented Generation.
          </p>
          <div className="flex justify-center gap-4 pt-4">
            <Link
              to="/chat"
              className="inline-flex items-center gap-2 bg-primary text-primary-foreground px-6 py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors shadow-sm"
            >
              Start Chatting
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              to="/about"
              className="inline-flex items-center gap-2 bg-secondary text-secondary-foreground px-6 py-3 rounded-lg font-medium hover:bg-secondary/80 transition-colors"
            >
              Learn More
            </Link>
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-card border border-border p-6 rounded-xl shadow-sm text-center space-y-4">
            <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center mx-auto dark:bg-blue-900/30 dark:text-blue-400">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold text-foreground">Evidence Based</h3>
            <p className="text-sm text-muted-foreground">
              Answers are grounded in a vetted medical knowledge base to reduce hallucinations.
            </p>
          </div>

          <div className="bg-card border border-border p-6 rounded-xl shadow-sm text-center space-y-4">
            <div className="w-12 h-12 bg-green-100 text-green-600 rounded-lg flex items-center justify-center mx-auto dark:bg-green-900/30 dark:text-green-400">
              <Database className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold text-foreground">Context Aware</h3>
            <p className="text-sm text-muted-foreground">
              Utilises FAISS semantic search to retrieve the most relevant literature before answering.
            </p>
          </div>

          <div className="bg-card border border-border p-6 rounded-xl shadow-sm text-center space-y-4">
            <div className="w-12 h-12 bg-purple-100 text-purple-600 rounded-lg flex items-center justify-center mx-auto dark:bg-purple-900/30 dark:text-purple-400">
              <Zap className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold text-foreground">Gemma 2B Powered</h3>
            <p className="text-sm text-muted-foreground">
              Runs a highly efficient, fine-tuned LoRA adapter on Google's state-of-the-art Gemma architecture.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default LandingPage;
