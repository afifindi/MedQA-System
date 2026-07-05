import React from 'react';
import { Link } from 'react-router-dom';
import { FileQuestion, ArrowLeft } from 'lucide-react';

const NotFoundPage = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full bg-background p-6 text-center space-y-6">
      <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center">
        <FileQuestion className="w-10 h-10 text-muted-foreground" />
      </div>
      <h1 className="text-4xl font-bold tracking-tight text-foreground">404 - Page Not Found</h1>
      <p className="text-muted-foreground max-w-md">
        The page you are looking for does not exist or has been moved.
      </p>
      <Link
        to="/"
        className="inline-flex items-center gap-2 text-primary hover:text-primary/80 font-medium transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Return to Home
      </Link>
    </div>
  );
};

export default NotFoundPage;
