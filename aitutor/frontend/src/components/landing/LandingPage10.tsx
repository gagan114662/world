/**
 * Landing Page 10: Interactive Showcase
 * Hover effects, interactive elements
 */
import React, { useState } from 'react';
import TeachrLogo from './TeachrLogo';
import { landingFeatures, howItWorksSteps, benefits } from './landingData';

interface LandingPage10Props {
  onGetStarted: () => void;
}

const LandingPage10: React.FC<LandingPage10Props> = ({ onGetStarted }) => {
  const [hoveredFeature, setHoveredFeature] = useState<number | null>(null);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      color: '#2d3748',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
      position: 'relative'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 24px' }}>
        {/* Header */}
        <header style={{
          padding: '32px 0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <TeachrLogo size="medium" color="#2d3748" />
          <div style={{
            fontSize: '14px',
            fontWeight: 600,
            color: '#718096'
          }}>
            AI-Powered Adaptive Learning
          </div>
        </header>

        {/* Hero Section */}
        <section style={{
          padding: '100px 0',
          textAlign: 'center'
        }}>
          <h1 style={{
            fontSize: '68px',
            fontWeight: 800,
            lineHeight: 1.1,
            marginBottom: '32px',
            color: '#2d3748',
            letterSpacing: '-0.03em'
          }}>
            Learn Interactively with AI
          </h1>
          <p style={{
            fontSize: '22px',
            fontWeight: 400,
            lineHeight: 1.7,
            marginBottom: '48px',
            color: '#4a5568',
            maxWidth: '800px',
            margin: '0 auto 48px'
          }}>
            Teachr uses advanced AI to create a personalized learning experience. Our adaptive system analyzes your performance in real-time, adjusts question difficulty automatically, and provides instant help from our AI tutor. Track your progress, master skills at your own pace, and build confidence through intelligent practice.
          </p>
          <button
            onClick={onGetStarted}
            style={{
              padding: '20px 56px',
              fontSize: '20px',
              fontWeight: 600,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '50px',
              cursor: 'pointer',
              boxShadow: '0 10px 30px rgba(102,126,234,0.4)',
              transition: 'all 0.3s ease',
              position: 'relative',
              overflow: 'hidden'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-4px) scale(1.05)';
              e.currentTarget.style.boxShadow = '0 15px 40px rgba(102,126,234,0.5)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0) scale(1)';
              e.currentTarget.style.boxShadow = '0 10px 30px rgba(102,126,234,0.4)';
            }}
          >
            Start Learning
          </button>
        </section>

        {/* Interactive Image Section */}
        <section style={{
          margin: '80px 0',
          textAlign: 'center',
          position: 'relative'
        }}>
          <div
            style={{
              display: 'inline-block',
              borderRadius: '24px',
              overflow: 'hidden',
              boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
              transition: 'all 0.3s ease',
              cursor: 'pointer'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.02)';
              e.currentTarget.style.boxShadow = '0 25px 70px rgba(0,0,0,0.2)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
              e.currentTarget.style.boxShadow = '0 20px 60px rgba(0,0,0,0.15)';
            }}
          >
            <img
              src="/landing-screenshots/home-screen-placeholder.png"
              alt="Teachr Interface"
              style={{
                width: '100%',
                maxWidth: '900px',
                height: 'auto',
                display: 'block'
              }}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.parentElement!.innerHTML = `
                  <div style="
                    width: 100%;
                    max-width: 900px;
                    height: 500px;
                    background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #718096;
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

        {/* Interactive Features Grid */}
        <section style={{
          padding: '100px 0',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '32px'
        }}>
          {[
            { icon: '🎯', title: 'Smart Adaptation', desc: 'Questions adjust to your skill level', color: '#667eea' },
            { icon: '🤖', title: 'AI Support', desc: 'Get help from our intelligent tutor', color: '#764ba2' },
            { icon: '📊', title: 'Track Progress', desc: 'Monitor your improvement over time', color: '#f093fb' },
            { icon: '✨', title: 'Engaging Practice', desc: 'Fun questions that keep you motivated', color: '#4facfe' }
          ].map((feature, idx) => (
            <div
              key={idx}
              style={{
                background: '#FFFFFF',
                borderRadius: '20px',
                padding: '40px',
                textAlign: 'center',
                boxShadow: hoveredFeature === idx 
                  ? '0 20px 60px rgba(0,0,0,0.15)' 
                  : '0 4px 20px rgba(0,0,0,0.08)',
                transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                transform: hoveredFeature === idx ? 'translateY(-12px) scale(1.02)' : 'translateY(0) scale(1)',
                border: hoveredFeature === idx ? `3px solid ${feature.color}` : '3px solid transparent',
                cursor: 'pointer'
              }}
              onMouseEnter={() => setHoveredFeature(idx)}
              onMouseLeave={() => setHoveredFeature(null)}
            >
              <div style={{
                fontSize: '64px',
                marginBottom: '24px',
                transform: hoveredFeature === idx ? 'scale(1.2) rotate(5deg)' : 'scale(1) rotate(0deg)',
                transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                display: 'inline-block'
              }}>
                {feature.icon}
              </div>
              <h3 style={{
                fontSize: '24px',
                fontWeight: 700,
                marginBottom: '16px',
                color: '#2d3748',
                transform: hoveredFeature === idx ? 'translateY(-4px)' : 'translateY(0)',
                transition: 'all 0.4s ease'
              }}>
                {feature.title}
              </h3>
              <p style={{
                fontSize: '16px',
                lineHeight: 1.6,
                color: '#718096',
                opacity: hoveredFeature === idx ? 1 : 0.8,
                transition: 'opacity 0.3s ease'
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
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '32px',
          margin: '60px 0',
          boxShadow: '0 20px 60px rgba(102,126,234,0.3)',
          color: '#FFFFFF'
        }}>
          <h2 style={{
            fontSize: '56px',
            fontWeight: 800,
            marginBottom: '24px',
            color: '#FFFFFF',
            letterSpacing: '-0.02em'
          }}>
            Ready to Master New Skills?
          </h2>
          <p style={{
            fontSize: '22px',
            marginBottom: '48px',
            color: 'rgba(255,255,255,0.9)',
            maxWidth: '700px',
            margin: '0 auto 48px'
          }}>
            Join thousands of students who are mastering new skills and building confidence every day.
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
          borderTop: '1px solid rgba(0,0,0,0.1)',
          color: '#718096',
          fontSize: '14px'
        }}>
          <p>© 2024 Teachr. Making learning accessible for everyone.</p>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage10;

