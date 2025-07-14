import { Link } from 'react-router-dom';

export const Header = () => {
  return (
    <header className="flex items-center justify-between h-16 px-4 bg-white border-b border-gray-200">
      <div className="text-lg font-medium text-gray-800">AIRI Assistant</div>
      <nav className="space-x-4 text-sm text-gray-600">
        <Link
          to="/"
          className="hover:text-black transition-colors"
        >
          Home
        </Link>
        <Link
          to="/chat"
          className="hover:text-black transition-colors"
        >
          Chat
        </Link>
      </nav>
    </header>
  );
};