/**
 * Landing Page Wrapper
 * Displays the neo-brutalism landing page
 */
import React, { useEffect } from 'react';
import { useHistory } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import LandingPageNeo from './LandingPageNeo';

// Commented out old landing page imports for future use
// import LandingPage1 from './LandingPage1';
// import LandingPage2 from './LandingPage2';
// import LandingPage3 from './LandingPage3';
// import LandingPage4 from './LandingPage4';
// import LandingPage5 from './LandingPage5';
// import LandingPage6 from './LandingPage6';
// import LandingPage7 from './LandingPage7';
// import LandingPage8 from './LandingPage8';
// import LandingPage9 from './LandingPage9';
// import LandingPage10 from './LandingPage10';

const LandingPageWrapper: React.FC = () => {
  const history = useHistory();
  const { isAuthenticated, isLoading } = useAuth();

  // Commented out random landing page selection logic
  // const [selectedPage, setSelectedPage] = useState<number | null>(null);

  // Get random landing page (persist for session) - COMMENTED OUT
  // const getRandomLandingPage = (): number => {
  //   // Check sessionStorage first (persist for session)
  //   const stored = sessionStorage.getItem('landingPageIndex');
  //   if (stored) {
  //     return parseInt(stored, 10);
  //   }
  //   
  //   // Generate random number 1-10
  //   const randomIndex = Math.floor(Math.random() * 10) + 1;
  //   sessionStorage.setItem('landingPageIndex', randomIndex.toString());
  //   return randomIndex;
  // };

  useEffect(() => {
    // If authenticated, redirect to app
    if (!isLoading && isAuthenticated) {
      history.replace('/app');
      return;
    }
  }, [isAuthenticated, isLoading, history]);

  // Show loading while checking authentication
  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        background: '#FFFDF5'
      }}>
        <div>Loading...</div>
      </div>
    );
  }

  // Render landing page
  const handleGetStarted = () => {
    history.push('/app/login');
  };

  // Commented out switch statement for old random selection
  // switch (selectedPage) {
  //   case 1:
  //     return <LandingPage1 onGetStarted={handleGetStarted} />;
  //   case 2:
  //     return <LandingPage2 onGetStarted={handleGetStarted} />;
  //   case 3:
  //     return <LandingPage3 onGetStarted={handleGetStarted} />;
  //   case 4:
  //     return <LandingPage4 onGetStarted={handleGetStarted} />;
  //   case 5:
  //     return <LandingPage5 onGetStarted={handleGetStarted} />;
  //   case 6:
  //     return <LandingPage6 onGetStarted={handleGetStarted} />;
  //   case 7:
  //     return <LandingPage7 onGetStarted={handleGetStarted} />;
  //   case 8:
  //     return <LandingPage8 onGetStarted={handleGetStarted} />;
  //   case 9:
  //     return <LandingPage9 onGetStarted={handleGetStarted} />;
  //   case 10:
  //     return <LandingPage10 onGetStarted={handleGetStarted} />;
  //   default:
  //     return <LandingPage1 onGetStarted={handleGetStarted} />;
  // }

  return <LandingPageNeo onGetStarted={handleGetStarted} />;
};

export default LandingPageWrapper;

