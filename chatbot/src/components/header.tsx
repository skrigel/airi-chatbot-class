import {Link } from 'react-router-dom'

export const Header = () => {
  return (
    <>
      <header className="flex items-center justify-between h-20 px-2 sm:px-4 py-2 bg-red-500 text-black dark:text-white w-full">
      <nav className='text-lg space-x-5'>
      <Link to="/">Home</Link>
      <Link to="/chat">Chat</Link>
      </nav>

      </header>
    </>
  );
};