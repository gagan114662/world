/**
 * Landing Page 8: Animated Hero
 * Subtle animations, engaging visuals
 */
import React, { useEffect, useState } from 'react';
import TeachrLogo from './TeachrLogo';
import { landingFeatures, howItWorksSteps, benefits } from './landingData';

interface LandingPage8Props {
  onGetStarted: () => void;
}

const LandingPage8: React.FC<LandingPage8Props> = ({ onGetStarted }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: '#FFFFFF',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Animated background circles */}
      <div style={{
        position: 'absolute',
        top: '10%',
        left: '10%',
        width: '300px',
        height: '300px',
        background: 'rgba(255,255,255,0.1)',
        borderRadius: '50%',
        filter: 'blur(60px)',
        animation: 'float 20s ease-in-out infinite'
      }} />
      <div style={{
        position: 'absolute',
        bottom: '10%',
        right: '10%',
        width: '400px',
        height: '400px',
        background: 'rgba(255,255,255,0.1)',
        borderRadius: '50%',
        filter: 'blur(60px)',
        animation: 'float 25s ease-in-out infinite reverse'
      }} />

      <style>{`
        @keyframes float {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(30px, -30px) scale(1.1); }
        }
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 24px', position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <header style={{
          padding: '32px 0',
          opacity: isVisible ? 1 : 0,
          animation: isVisible ? 'fadeInUp 0.6s ease-out' : 'none'
        }}>
          <TeachrLogo size="medium" color="#FFFFFF" />
        </header>

        {/* Hero Section */}
        <section style={{
          padding: '120px 0',
          textAlign: 'center'
        }}>
          <div style={{
            opacity: isVisible ? 1 : 0,
            animation: isVisible ? 'fadeInUp 0.8s ease-out 0.2s both' : 'none'
          }}>
            <h1 style={{
              fontSize: '72px',
              fontWeight: 900,
              lineHeight: 1.1,
              marginBottom: '32px',
              color: '#FFFFFF',
              letterSpacing: '-0.03em',
              textShadow: '0 4px 20px rgba(0,0,0,0.2)'
            }}>
              Learn Smarter with AI
            </h1>
            <p style={{
              fontSize: '24px',
              fontWeight: 400,
              lineHeight: 1.7,
              marginBottom: '48px',
              color: 'rgba(255,255,255,0.9)',
              maxWidth: '800px',
              margin: '0 auto 48px'
            }}>
              Teachr uses cutting-edge AI to create a personalized learning experience. Our adaptive system analyzes your performance in real-time, adjusts question difficulty automatically, and provides instant help from our AI tutor. Track your progress, master skills at your own pace, and build confidence through intelligent practice.
            </p>
            <button
              onClick={onGetStarted}
              style={{
                padding: '20px 56px',
                fontSize: '20px',
                fontWeight: 600,
                background: '#FFFFFF',
                color: '#667eea',
                border: 'none',
                borderRadius: '50px',
                cursor: 'pointer',
                boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
                transition: 'all 0.3s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px) scale(1.05)';
                e.currentTarget.style.boxShadow = '0 15px 50px rgba(0,0,0,0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.boxShadow = '0 10px 40px rgba(0,0,0,0.2)';
              }}
            >
              Get Started
            </button>
          </div>
        </section>

        {/* Image Section */}
        <section style={{
          margin: '80px 0',
          textAlign: 'center',
          opacity: isVisible ? 1 : 0,
          animation: isVisible ? 'fadeInUp 1s ease-out 0.4s both' : 'none'
        }}>
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            backdropFilter: 'blur(20px)',
            borderRadius: '32px',
            padding: '32px',
            display: 'inline-block',
            border: '1px solid rgba(255,255,255,0.2)',
            boxShadow: '0 20px 60px rgba(0,0,0,0.2)'
          }}>
            <img
              src="/landing-screenshots/home-screen-placeholder.png"
              alt="Teachr Interface"
              style={{
                width: '100%',
                maxWidth: '900px',
                height: 'auto',
                borderRadius: '24px',
                boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
              }}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.parentElement!.innerHTML = `
                  <div style="
                    width: 100%;
                    max-width: 900px;
                    height: 500px;
                    background: rgba(255,255,255,0.15);
                    border-radius: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    color: white;
                    font-size: 18px;
                    font-weight: 600;
                  ">
                    Learning Interface Preview
                  </div>
                `;
              }}
            />
          </div>
        </section>

        {/* Features Section */}
        <section style={{
          padding: '100px 0',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '40px'
        }}>
          {[
            { icon: '🎯', title: 'Smart Learning', desc: 'Questions adapt to your level' },
            { icon: '🤖', title: 'AI Assistance', desc: 'Get help whenever you need it' },
            { icon: '📊', title: 'Track Progress', desc: 'See improvement over time' },
            { icon: '✨', title: 'Fun Practice', desc: 'Engaging questions keep you motivated' }
          ].map((feature, idx) => (
            <div
              key={idx}
              style={{
                background: 'rgba(255,255,255,0.1)',
                backdropFilter: 'blur(10px)',
                padding: '40px',
                borderRadius: '24px',
                textAlign: 'center',
                border: '1px solid rgba(255,255,255,0.2)',
                transition: 'all 0.3s ease',
                opacity: isVisible ? 1 : 0,
                animation: isVisible ? `fadeInUp 0.6s ease-out ${0.6 + idx * 0.1}s both` : 'none'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-12px)';
                e.currentTarget.style.background = 'rgba(255,255,255,0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
              }}
            >
              <div style={{ fontSize: '56px', marginBottom: '24px' }}>{feature.icon}</div>
              <h3 style={{
                fontSize: '24px',
                fontWeight: 700,
                marginBottom: '16px',
                color: '#FFFFFF'
              }}>
                {feature.title}
              </h3>
              <p style={{
                fontSize: '16px',
                lineHeight: 1.6,
                color: 'rgba(255,255,255,0.9)'
              }}>
                {feature.desc}
              </p>
            </div>
          ))}
        </section>

        {/* CTA Section */}
        <section style={{
          padding: '100px 0',
          textAlign: 'center',
          background: 'rgba(255,255,255,0.1)',
          backdropFilter: 'blur(20px)',
          borderRadius: '40px',
          margin: '60px 0',
          border: '1px solid rgba(255,255,255,0.2)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.2)'
        }}>
          <h2 style={{
            fontSize: '56px',
            fontWeight: 900,
            marginBottom: '24px',
            color: '#FFFFFF',
            textShadow: '0 4px 20px rgba(0,0,0,0.2)'
          }}>
            Ready to Excel?
          </h2>
          <p style={{
            fontSize: '22px',
            color: 'rgba(255,255,255,0.9)',
            marginBottom: '48px'
          }}>
            Join thousands of students building confidence every day.
          </p>
          <button
            onClick={onGetStarted}
            style={{
              padding: '22px 64px',
              fontSize: '22px',
              fontWeight: 600,
              background: '#FFFFFF',
              color: '#667eea',
              border: 'none',
              borderRadius: '50px',
              cursor: 'pointer',
              boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
              transition: 'all 0.3s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-4px) scale(1.05)';
              e.currentTarget.style.boxShadow = '0 15px 50px rgba(0,0,0,0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0) scale(1)';
              e.currentTarget.style.boxShadow = '0 10px 40px rgba(0,0,0,0.2)';
            }}
          >
            Start Practicing
          </button>
        </section>

        {/* Footer */}
        <footer style={{
          padding: '40px 0',
          textAlign: 'center',
          color: 'rgba(255,255,255,0.7)',
          fontSize: '14px'
        }}>
          <p>© 2024 Teachr. Making learning accessible for everyone.</p>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage8;

