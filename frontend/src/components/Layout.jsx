import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Stethoscope, Info, MessageSquare } from 'lucide-react';

const Layout = () => {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar Navigation */}
      <nav className="w-64 border-r border-border bg-card flex flex-col transition-all duration-300">
        <div className="p-4 border-b border-border flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg text-primary">
            <Stethoscope className="w-6 h-6" />
          </div>
          <h1 className="font-semibold text-lg tracking-tight">MedQA System</h1>
        </div>

        <div className="flex-1 py-4 flex flex-col gap-2 px-3">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                isActive ? 'bg-secondary text-secondary-foreground font-medium' : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'
              }`
            }
          >
            <Stethoscope className="w-5 h-5" />
            <span>Home</span>
          </NavLink>
          
          <NavLink
            to="/chat"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                isActive ? 'bg-secondary text-secondary-foreground font-medium' : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'
              }`
            }
          >
            <MessageSquare className="w-5 h-5" />
            <span>Medical Chat</span>
          </NavLink>

          <NavLink
            to="/about"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                isActive ? 'bg-secondary text-secondary-foreground font-medium' : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'
              }`
            }
          >
            <Info className="w-5 h-5" />
            <span>About</span>
          </NavLink>
        </div>
        
        <div className="p-4 border-t border-border text-xs text-muted-foreground">
          <p>Powered by Gemma-2B & LoRA</p>
          <p>Retrieval Augmented Generation</p>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
