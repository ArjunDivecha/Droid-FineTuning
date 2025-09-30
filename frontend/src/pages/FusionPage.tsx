import React from 'react';
import { Sparkles } from 'lucide-react';

const FusionPage: React.FC = () => {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <Sparkles className="w-16 h-16 mx-auto mb-4 text-primary-500" />
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Adapter Fusion
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Fusion functionality coming soon
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
          Blend multiple adapters to create hybrid models
        </p>
      </div>
    </div>
  );
};

export default FusionPage;
