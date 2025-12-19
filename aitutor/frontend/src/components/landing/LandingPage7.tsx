/**
 * Landing Page 7: Split-Screen Design
 * Image/content split, modern layout
 */
import React from 'react';
import TeachrLogo from './TeachrLogo';
import { landingFeatures, howItWorksSteps, benefits } from './landingData';

interface LandingPage7Props {
  onGetStarted: () => void;
}

const LandingPage7: React.FC<LandingPage7Props> = ({ onGetStarted }) => {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#FFFFFF',
      color: '#1a1a1a',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif'
    }}>
      {/* Header */}
      <header style={{
        padding: '24px 0',
        borderBottom: '1px solid #e5e7eb',
        position: 'sticky',
        top: 0,
        background: 'rgba(255,255,255,0.95)',
        backdropFilter: 'blur(10px)',
        zIndex: 100
      }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 24px' }}>
          <TeachrLogo size="medium" color="#1a1a1a" />
        </div>
      </header>

      {/* Split Hero Section */}
      <section style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        minHeight: 'calc(100vh - 80px)',
        alignItems: 'center'
      }}>
        {/* Left: Content */}
        <div style={{
          padding: '80px 60px',
          background: '#FFFFFF'
        }}>
          <div style={{ maxWidth: '500px' }}>
            <h1 style={{
              fontSize: '64px',
              fontWeight: 800,
              lineHeight: 1.1,
              marginBottom: '24px',
              color: '#1a1a1a',
              letterSpacing: '-0.02em'
            }}>
              Learning That Adapts to You
            </h1>
            <p style={{
              fontSize: '22px',
              fontWeight: 400,
              lineHeight: 1.7,
              marginBottom: '48px',
              color: '#4b5563'
            }}>
              Teachr uses advanced AI to create a personalized learning experience. Our adaptive system analyzes your performance in real-time, adjusts question difficulty automatically, and provides instant help from our AI tutor. Track your progress, master skills at your own pace, and build confidence through intelligent practice.
            </p>
            <button
              onClick={onGetStarted}
              style={{
                padding: '18px 40px',
                fontSize: '18px',
                fontWeight: 600,
                background: '#1a1a1a',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '12px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#374151';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#1a1a1a';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              Learn More
            </button>
          </div>
        </div>

        {/* Right: Image */}
        <div style={{
          background: 'linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)',
          padding: '60px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative'
        }}>
          <div style={{
            width: '100%',
            maxWidth: '600px',
            borderRadius: '24px',
            overflow: 'hidden',
            boxShadow: '0 20px 60px rgba(0,0,0,0.1)'
          }}>
            <img
              src="/landing-screenshots/home-screen-placeholder.png"
              alt="Teachr Interface"
              style={{
                width: '100%',
                height: 'auto',
                display: 'block'
              }}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.parentElement!.innerHTML = `
                  <div style="
                    width: 100%;
                    height: 500px;
                    background: linear-gradient(135deg, #d1d5db 0%, #9ca3af 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #4b5563;
                    font-size: 18px;
                    font-weight: 500;
                  ">
                    Learning Interface Preview
                  </div>
                `;
              }}
            />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section style={{
        padding: '100px 0',
        background: '#f9fafb'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 24px' }}>
          <div style={{
            textAlign: 'center',
            marginBottom: '60px'
          }}>
            <h2 style={{
              fontSize: '48px',
              fontWeight: 700,
              marginBottom: '16px',
              color: '#1a1a1a'
            }}>
              Why Choose Teachr?
            </h2>
            <p style={{
              fontSize: '20px',
              color: '#6b7280'
            }}>
              Everything you need to succeed in your learning journey
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '40px'
          }}>
            {[
              { icon: '🎯', title: 'Adaptive Learning', desc: 'Questions automatically adjust to match your current skill level' },
              { icon: '🤖', title: 'AI-Powered Support', desc: 'Get instant help from our intelligent tutor whenever you need it' },
              { icon: '📊', title: 'Progress Tracking', desc: 'Monitor your improvement with detailed analytics and insights' },
              { icon: '✨', title: 'Engaging Practice', desc: 'Interactive questions that make learning fun and enjoyable' }
            ].map((feature, idx) => (
              <div
                key={idx}
                style={{
                  background: '#FFFFFF',
                  padding: '40px',
                  borderRadius: '16px',
                  boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 24px rgba(0,0,0,0.06)';
                }}
              >
                <div style={{ fontSize: '40px', marginBottom: '16px' }}>{feature.icon}</div>
                <h3 style={{
                  fontSize: '24px',
                  fontWeight: 700,
                  marginBottom: '12px',
                  color: '#1a1a1a'
                }}>
                  {feature.title}
                </h3>
                <p style={{
                  fontSize: '16px',
                  lineHeight: 1.6,
                  color: '#6b7280'
                }}>
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section style={{
        padding: '100px 0',
        textAlign: 'center',
        background: '#1a1a1a',
        color: '#FFFFFF'
      }}>
        <h2 style={{
          fontSize: '56px',
          fontWeight: 800,
          marginBottom: '24px',
          color: '#FFFFFF'
        }}>
          Start Learning Today
        </h2>
        <p style={{
          fontSize: '22px',
          marginBottom: '40px',
          color: 'rgba(255,255,255,0.8)'
        }}>
          Join thousands of students who are mastering new skills and building confidence every day.
        </p>
        <button
          onClick={onGetStarted}
          style={{
            padding: '20px 56px',
            fontSize: '20px',
            fontWeight: 600,
            background: '#FFFFFF',
            color: '#1a1a1a',
            border: 'none',
            borderRadius: '12px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = '#f3f4f6';
            e.currentTarget.style.transform = 'translateY(-2px)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = '#FFFFFF';
            e.currentTarget.style.transform = 'translateY(0)';
          }}
        >
          Start Today
        </button>
      </section>

      {/* Footer */}
      <footer style={{
        padding: '40px 0',
        textAlign: 'center',
        borderTop: '1px solid #e5e7eb',
        color: '#6b7280',
        fontSize: '14px'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 24px' }}>
          <p>© 2024 Teachr. Making learning accessible for everyone.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage7;

