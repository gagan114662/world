/**
 * Landing Page 2: Minimalist Design - Enhanced
 * Clean, elegant, lots of white space with detailed content
 */
import React from 'react';
import TeachrLogo from './TeachrLogo';

interface LandingPage2Props {
  onGetStarted: () => void;
}

const LandingPage2: React.FC<LandingPage2Props> = ({ onGetStarted }) => {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#FFFFFF',
      color: '#1a1a1a',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif'
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 40px' }}>
        {/* Header */}
        <header style={{
          padding: '40px 0',
          borderBottom: '1px solid #e5e5e5',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <TeachrLogo size="medium" color="#1a1a1a" />
          <nav style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
            <a href="#features" style={{ color: '#666666', textDecoration: 'none', fontSize: '15px', fontWeight: 500 }}>Features</a>
            <a href="#how-it-works" style={{ color: '#666666', textDecoration: 'none', fontSize: '15px', fontWeight: 500 }}>How It Works</a>
            <button
              onClick={onGetStarted}
              style={{
                padding: '12px 28px',
                fontSize: '15px',
                fontWeight: 500,
                background: '#1a1a1a',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#333333';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#1a1a1a';
              }}
            >
              Get Started
            </button>
          </nav>
        </header>

        {/* Hero Section */}
        <section style={{
          padding: '140px 0 100px',
          textAlign: 'center',
          maxWidth: '900px',
          margin: '0 auto'
        }}>
          <div style={{
            display: 'inline-block',
            padding: '6px 16px',
            background: '#f5f5f5',
            borderRadius: '20px',
            fontSize: '13px',
            fontWeight: 600,
            color: '#666666',
            marginBottom: '32px',
            letterSpacing: '0.5px'
          }}>
            AI-Powered Adaptive Learning Platform
          </div>
          <h1 style={{
            fontSize: '72px',
            fontWeight: 700,
            lineHeight: 1.1,
            marginBottom: '32px',
            letterSpacing: '-0.03em',
            color: '#1a1a1a'
          }}>
            Personalized Learning That Adapts to You
          </h1>
          <p style={{
            fontSize: '22px',
            fontWeight: 400,
            lineHeight: 1.7,
            marginBottom: '48px',
            color: '#666666',
            maxWidth: '700px',
            margin: '0 auto 48px'
          }}>
            Teachr uses advanced AI to create a personalized learning experience. Our adaptive system analyzes your performance, adjusts question difficulty in real-time, and provides instant help from our AI tutor. Track your progress, master skills at your own pace, and build confidence through practice.
          </p>
          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={onGetStarted}
              style={{
                padding: '18px 40px',
                fontSize: '17px',
                fontWeight: 600,
                background: '#1a1a1a',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#333333';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#1a1a1a';
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
              }}
            >
              Start Learning Free
            </button>
            <button
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              style={{
                padding: '18px 40px',
                fontSize: '17px',
                fontWeight: 600,
                background: 'transparent',
                color: '#1a1a1a',
                border: '2px solid #e5e5e5',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#1a1a1a';
                e.currentTarget.style.background = '#f9f9f9';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#e5e5e5';
                e.currentTarget.style.background = 'transparent';
              }}
            >
              Learn More
            </button>
          </div>
        </section>

        {/* Image Section */}
        <section style={{ margin: '100px 0', textAlign: 'center' }}>
          <div style={{
            borderRadius: '16px',
            overflow: 'hidden',
            boxShadow: '0 20px 60px rgba(0,0,0,0.08)',
            maxWidth: '1200px',
            margin: '0 auto'
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
                    height: 600px;
                    background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #999;
                    font-size: 18px;
                  ">
                    Learning Interface Preview
                  </div>
                `;
              }}
            />
          </div>
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" style={{
          padding: '120px 0',
          background: '#fafafa'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '80px' }}>
            <h2 style={{
              fontSize: '56px',
              fontWeight: 700,
              marginBottom: '20px',
              color: '#1a1a1a',
              letterSpacing: '-0.02em'
            }}>
              How It Works
            </h2>
            <p style={{
              fontSize: '20px',
              color: '#666666',
              maxWidth: '600px',
              margin: '0 auto'
            }}>
              A simple, effective learning process designed for your success
            </p>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '60px',
            maxWidth: '1200px',
            margin: '0 auto'
          }}>
            {[
              {
                step: '1',
                title: 'Create Your Profile',
                description: 'Sign up with Google in seconds. Tell us your grade level, and our system automatically personalizes your learning experience. No complex setup required.'
              },
              {
                step: '2',
                title: 'Start Practicing',
                description: 'Begin with questions matched to your skill level. Our adaptive AI analyzes each answer and adjusts difficulty in real-time, ensuring optimal challenge.'
              },
              {
                step: '3',
                title: 'Get Help & Improve',
                description: 'Stuck? Ask our AI tutor for instant explanations. Track your progress with detailed analytics and watch your skills improve over time.'
              }
            ].map((item, idx) => (
              <div key={idx}>
                <div style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: '#1a1a1a',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px',
                  fontWeight: 700,
                  marginBottom: '32px'
                }}>
                  {item.step}
                </div>
                <h3 style={{
                  fontSize: '26px',
                  fontWeight: 600,
                  marginBottom: '16px',
                  color: '#1a1a1a'
                }}>
                  {item.title}
                </h3>
                <p style={{
                  fontSize: '17px',
                  lineHeight: 1.7,
                  color: '#666666'
                }}>
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Features Section */}
        <section id="features" style={{
          padding: '120px 0'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '80px' }}>
            <h2 style={{
              fontSize: '56px',
              fontWeight: 700,
              marginBottom: '20px',
              color: '#1a1a1a',
              letterSpacing: '-0.02em'
            }}>
              Everything You Need to Succeed
            </h2>
            <p style={{
              fontSize: '20px',
              color: '#666666',
              maxWidth: '600px',
              margin: '0 auto'
            }}>
              Powerful features designed to enhance your learning experience
            </p>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '60px',
            maxWidth: '1200px',
            margin: '0 auto'
          }}>
            {[
              {
                icon: '🎯',
                title: 'Adaptive Learning Engine',
                description: 'Our intelligent system uses advanced algorithms to analyze your performance patterns. Questions automatically adjust in difficulty based on your answers, ensuring you\'re always challenged at the right level. The system learns your strengths and weaknesses, creating a personalized learning path unique to you.'
              },
              {
                icon: '🤖',
                title: 'AI Tutor with Real-Time Help',
                description: 'Powered by Google Gemini, our AI tutor provides instant assistance whenever you need it. Ask questions via voice or text, receive step-by-step explanations, and get personalized guidance. The tutor understands context, remembers your learning history, and adapts explanations to your level.'
              },
              {
                icon: '📊',
                title: 'Comprehensive Progress Analytics',
                description: 'Track your improvement with detailed metrics including memory strength for each skill, practice frequency, accuracy rates, and time spent. Visualize your learning journey with progress charts and identify areas that need more attention. See how you improve over days, weeks, and months.'
              },
              {
                icon: '✨',
                title: 'Rich Interactive Questions',
                description: 'Engage with diverse question formats including multiple choice, numeric input, graphing exercises, drag-and-drop matching, expression input, and more. Each question type is designed to test different skills and keep learning engaging and effective.'
              },
              {
                icon: '🎓',
                title: 'Grade-Appropriate Content',
                description: 'Content automatically matches your grade level from Kindergarten through 12th grade. The system ensures you\'re always practicing at the appropriate level for your age and learning stage, with smooth progression as you advance.'
              },
              {
                icon: '⚡',
                title: 'Instant Feedback System',
                description: 'Get immediate feedback on every answer with detailed explanations. Learn from mistakes in real-time, understand concepts better through corrections, and access hints when needed. Feedback is personalized to help you understand not just what was wrong, but why.'
              }
            ].map((feature, idx) => (
              <div
                key={idx}
                style={{
                  padding: '40px',
                  background: '#FFFFFF',
                  borderRadius: '12px',
                  border: '1px solid #e5e5e5',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = '0 12px 40px rgba(0,0,0,0.08)';
                  e.currentTarget.style.borderColor = '#1a1a1a';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                  e.currentTarget.style.borderColor = '#e5e5e5';
                }}
              >
                <div style={{ fontSize: '48px', marginBottom: '24px' }}>{feature.icon}</div>
                <h3 style={{
                  fontSize: '24px',
                  fontWeight: 600,
                  marginBottom: '16px',
                  color: '#1a1a1a'
                }}>
                  {feature.title}
                </h3>
                <p style={{
                  fontSize: '16px',
                  lineHeight: 1.7,
                  color: '#666666'
                }}>
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Benefits Section */}
        <section style={{
          padding: '100px 0',
          background: '#1a1a1a',
          color: '#FFFFFF',
          margin: '80px 0',
          borderRadius: '16px'
        }}>
          <div style={{ maxWidth: '1000px', margin: '0 auto', textAlign: 'center' }}>
            <h2 style={{
              fontSize: '48px',
              fontWeight: 700,
              marginBottom: '40px',
              color: '#FFFFFF',
              letterSpacing: '-0.02em'
            }}>
              Why Students Choose Teachr
            </h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '32px',
              textAlign: 'left'
            }}>
              {[
                'Practice at your own pace with no pressure or deadlines',
                'Questions adapt automatically to your skill level',
                'Get instant help from AI tutor 24/7',
                'Track detailed progress and see improvement over time',
                'Learn from mistakes with comprehensive explanations',
                'Engaging interactive content keeps you motivated'
              ].map((benefit, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '16px',
                    padding: '24px',
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.1)'
                  }}
                >
                  <div style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: '#FFFFFF',
                    flexShrink: 0,
                    marginTop: '2px'
                  }} />
                  <p style={{
                    fontSize: '17px',
                    fontWeight: 400,
                    margin: 0,
                    color: 'rgba(255,255,255,0.9)',
                    lineHeight: 1.6
                  }}>
                    {benefit}
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
          background: '#fafafa',
          borderRadius: '16px',
          margin: '60px 0'
        }}>
          <h2 style={{
            fontSize: '48px',
            fontWeight: 700,
            marginBottom: '24px',
            color: '#1a1a1a',
            letterSpacing: '-0.02em'
          }}>
            Start Your Learning Journey Today
          </h2>
          <p style={{
            fontSize: '20px',
            color: '#666666',
            marginBottom: '40px',
            maxWidth: '600px',
            margin: '0 auto 40px'
          }}>
            Join thousands of students who are mastering new skills and building confidence every day.
          </p>
          <button
            onClick={onGetStarted}
            style={{
              padding: '20px 48px',
              fontSize: '18px',
              fontWeight: 600,
              background: '#1a1a1a',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#333333';
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,0,0,0.15)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#1a1a1a';
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
            }}
          >
            Begin Your Journey
          </button>
          <p style={{
            fontSize: '14px',
            color: '#999999',
            marginTop: '24px'
          }}>
            Free to start • No credit card required
          </p>
        </section>

        {/* Footer */}
        <footer style={{
          padding: '60px 0',
          borderTop: '1px solid #e5e5e5',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '24px'
        }}>
          <p style={{ margin: 0, color: '#999999', fontSize: '14px' }}>
            © 2024 Teachr. Making learning accessible for everyone.
          </p>
          <div style={{ display: 'flex', gap: '32px' }}>
            <a href="#" style={{ color: '#666666', textDecoration: 'none', fontSize: '14px' }}>Privacy</a>
            <a href="#" style={{ color: '#666666', textDecoration: 'none', fontSize: '14px' }}>Terms</a>
            <a href="#" style={{ color: '#666666', textDecoration: 'none', fontSize: '14px' }}>Contact</a>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage2;
