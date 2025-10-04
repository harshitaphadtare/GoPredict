import { Github, Rocket } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-slate-50 dark:bg-gray-800/50 border-t border-slate-200 dark:border-white/10 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4 py-5">
          <div className="flex items-center gap-2 text-sm font-medium text-slate-600 dark:text-slate-400">
            <Rocket className="h-4 w-4" />
            <span>
              Engineered by{' '}
              <a
                href="https://www.linkedin.com/in/harshitaphadtare/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 dark:text-blue-400 hover:underline"
              >
                Harshita Phadtare
              </a>
            </span>
          </div>
          <a
            href="https://github.com/harshitaphadtare/GoPredict"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm font-medium text-slate-600 dark:text-slate-400 transition-colors hover:text-blue-600 dark:hover:text-blue-400"
          >
            <Github className="h-4 w-4" />
            <span>Contribute on GitHub</span>
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
