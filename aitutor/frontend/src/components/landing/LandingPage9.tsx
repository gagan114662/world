/**
 * Landing Page 9: Magazine Style
 * Editorial layout, bold typography
 */
import React from 'react';
import TeachrLogo from './TeachrLogo';
import { landingFeatures, howItWorksSteps, benefits } from './landingData';

interface LandingPage9Props {
  onGetStarted: () => void;
}

const LandingPage9: React.FC<LandingPage9Props> = ({ onGetStarted }) => {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#FFFFFF',
      color: '#1a1a1a',
      fontFamily: '"Georgia", "Times New Roman", serif'
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 24px' }}>
        {/* Header */}
        <header style={{
          padding: '40px 0',
          borderBottom: '2px solid #1a1a1a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <TeachrLogo size="medium" color="#1a1a1a" />
          <div style={{
            fontSize: '14px',
            fontWeight: 600,
            letterSpacing: '0.2em',
            textTransform: 'uppercase',
            color: '#666666'
          }}>
            AI-Powered Learning Platform
          </div>
        </header>

        {/* Hero Section - Magazine Style */}
        <section style={{
          padding: '100px 0',
          display: 'grid',
          gridTemplateColumns: '1.5fr 1fr',
          gap: '80px',
          alignItems: 'center'
        }}>
          <div>
            <div style={{
              fontSize: '14px',
              fontWeight: 600,
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
              color: '#666666',
              marginBottom: '24px'
            }}>
              Personalized Learning
            </div>
            <h1 style={{
              fontSize: '80px',
              fontWeight: 900,
              lineHeight: 1.1,
              marginBottom: '32px',
              color: '#1a1a1a',
              letterSpacing: '-0.02em',
              fontFamily: '"Georgia", serif'
            }}>
              Master Skills Through Practice
            </h1>
            <p style={{
              fontSize: '22px',
              fontWeight: 400,
              lineHeight: 1.8,
              marginBottom: '48px',
              color: '#4b5563',
              fontFamily: '"Georgia", serif'
            }}>
              Every student learns differently. Our AI-powered platform adapts to your pace, providing personalized questions that challenge you just enough to grow. Get instant help from our AI tutor, track your progress with detailed analytics, and master any subject at your own pace.
            </p>
            <button
              onClick={onGetStarted}
              style={{
                padding: '18px 48px',
                fontSize: '16px',
                fontWeight: 700,
                background: '#1a1a1a',
                color: '#FFFFFF',
                border: '2px solid #1a1a1a',
                borderRadius: '0',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                fontFamily: 'inherit'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#FFFFFF';
                e.currentTarget.style.color = '#1a1a1a';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#1a1a1a';
                e.currentTarget.style.color = '#FFFFFF';
              }}
            >
              Start Learning
            </button>
          </div>

          <div style={{
            border: '2px solid #1a1a1a',
            padding: '8px',
            background: '#f9fafb'
          }}>
            <img
              src="/landing-screenshots/home-screen-placeholder.png"
              alt="Teachr Interface"
              style={{
                width: '100%',
                height: 'auto',
                display: 'block',
                border: '1px solid #e5e7eb'
              }}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.parentElement!.innerHTML = `
                  <div style="
                    width: 100%;
                    height: 500px;
                    background: #f3f4f6;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #6b7280;
                    font-size: 16px;
                    font-weight: 500;
                    border: 1px solid #e5e7eb;
                  ">
                    Learning Interface Preview
                  </div>
                `;
              }}
            />
          </div>
        </section>

        {/* Features - Editorial Style */}
        <section style={{
          padding: '120px 0',
          borderTop: '2px solid #1a1a1a',
          borderBottom: '2px solid #1a1a1a'
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '60px'
          }}>
            {[
              { icon: '🎯', title: 'Adaptive Questions', desc: 'Every question is carefully selected to match your current skill level, ensuring you\'re always learning at the right pace.' },
              { icon: '🤖', title: 'AI-Powered Help', desc: 'Stuck on a problem? Our intelligent tutor provides instant guidance, helping you understand concepts in real-time.' },
              { icon: '📊', title: 'Progress Insights', desc: 'Track your improvement with detailed analytics. See which skills you\'ve mastered and where you need more practice.' },
              { icon: '✨', title: 'Engaging Practice', desc: 'Learning doesn\'t have to be boring. Our interactive questions make practice enjoyable and motivating.' }
            ].map((feature, idx) => (
              <div key={idx} style={{ paddingRight: idx % 2 === 0 ? '40px' : '0' }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '24px'
                }}>
                  {feature.icon}
                </div>
                <h3 style={{
                  fontSize: '32px',
                  fontWeight: 700,
                  marginBottom: '16px',
                  color: '#1a1a1a',
                  fontFamily: '"Georgia", serif'
                }}>
                  {feature.title}
                </h3>
                <p style={{
                  fontSize: '18px',
                  lineHeight: 1.8,
                  color: '#4b5563',
                  fontFamily: '"Georgia", serif'
                }}>
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section style={{
          padding: '120px 0',
          textAlign: 'center',
          background: '#1a1a1a',
          color: '#FFFFFF',
          margin: '80px 0'
        }}>
          <h2 style={{
            fontSize: '64px',
            fontWeight: 900,
            marginBottom: '32px',
            color: '#FFFFFF',
            fontFamily: '"Georgia", serif',
            letterSpacing: '-0.02em'
          }}>
            Begin Your Learning Journey
          </h2>
          <p style={{
            fontSize: '24px',
            marginBottom: '48px',
            color: 'rgba(255,255,255,0.8)',
            fontFamily: '"Georgia", serif',
            maxWidth: '700px',
            margin: '0 auto 48px'
          }}>
            Join thousands of students who are mastering new skills and building confidence every day.
          </p>
          <button
            onClick={onGetStarted}
            style={{
              padding: '20px 56px',
              fontSize: '18px',
              fontWeight: 700,
              background: '#FFFFFF',
              color: '#1a1a1a',
              border: '2px solid #FFFFFF',
              borderRadius: '0',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              fontFamily: 'inherit'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = '#FFFFFF';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#FFFFFF';
              e.currentTarget.style.color = '#1a1a1a';
            }}
          >
            Get Started Now
          </button>
        </section>

        {/* Footer */}
        <footer style={{
          padding: '60px 0',
          textAlign: 'center',
          borderTop: '2px solid #1a1a1a',
          fontSize: '14px',
          color: '#666666',
          fontWeight: 600,
          letterSpacing: '0.1em',
          textTransform: 'uppercase'
        }}>
          <p>© 2024 Teachr. Making learning accessible for everyone.</p>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage9;

