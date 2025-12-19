/**
 * Landing Page 5: Dark Mode Elegant - Enhanced
 * Dark theme with neon accents, detailed content
 */
import React from 'react';
import TeachrLogo from './TeachrLogo';
import { landingFeatures, howItWorksSteps, benefits } from './landingData';

interface LandingPage5Props {
  onGetStarted: () => void;
}

const LandingPage5: React.FC<LandingPage5Props> = ({ onGetStarted }) => {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#0a0a0a',
      color: '#FFFFFF',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Animated grid background */}
      <div style={{
        position: 'absolute',
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(0,255,255,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0,255,255,0.03) 1px, transparent 1px)
        `,
        backgroundSize: '50px 50px',
        opacity: 0.5
      }} />
      {/* Glowing orbs */}
      <div style={{
        position: 'absolute',
        top: '20%',
        right: '10%',
        width: '400px',
        height: '400px',
        background: 'radial-gradient(circle, rgba(0,255,255,0.1) 0%, transparent 70%)',
        borderRadius: '50%',
        filter: 'blur(60px)'
      }} />
      <div style={{
        position: 'absolute',
        bottom: '20%',
        left: '10%',
        width: '300px',
        height: '300px',
        background: 'radial-gradient(circle, rgba(255,0,255,0.1) 0%, transparent 70%)',
        borderRadius: '50%',
        filter: 'blur(60px)'
      }} />

      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 40px', position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <header style={{
          padding: '40px 0',
          borderBottom: '1px solid rgba(0,255,255,0.2)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <TeachrLogo size="medium" color="#00FFFF" />
          <nav style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
            <a href="#features" style={{ color: 'rgba(0,255,255,0.8)', textDecoration: 'none', fontSize: '15px', fontWeight: 500 }}>Features</a>
            <button
              onClick={onGetStarted}
              style={{
                padding: '12px 28px',
                fontSize: '15px',
                fontWeight: 600,
                background: 'transparent',
                color: '#00FFFF',
                border: '2px solid #00FFFF',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#00FFFF';
                e.currentTarget.style.color = '#0a0a0a';
                e.currentTarget.style.boxShadow = '0 0 30px rgba(0,255,255,0.5)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = '#00FFFF';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              Get Started
            </button>
          </nav>
        </header>

        {/* Hero Section */}
        <section style={{
          padding: '120px 0',
          textAlign: 'center'
        }}>
          <div style={{
            marginBottom: '32px'
          }}>
            <span style={{
              display: 'inline-block',
              padding: '8px 20px',
              background: 'rgba(0,255,255,0.1)',
              border: '1px solid rgba(0,255,255,0.3)',
              borderRadius: '20px',
              fontSize: '13px',
              fontWeight: 600,
              color: '#00FFFF',
              letterSpacing: '0.1em',
              textTransform: 'uppercase'
            }}>
              AI-Powered Adaptive Learning Platform
            </span>
          </div>
          <h1 style={{
            fontSize: '80px',
            fontWeight: 900,
            lineHeight: 1.1,
            marginBottom: '32px',
            background: 'linear-gradient(135deg, #00FFFF 0%, #FF00FF 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            letterSpacing: '-0.03em'
          }}>
            Learn Smarter with AI
          </h1>
          <p style={{
            fontSize: '24px',
            fontWeight: 400,
            lineHeight: 1.7,
            marginBottom: '48px',
            color: 'rgba(255,255,255,0.7)',
            maxWidth: '800px',
            margin: '0 auto 48px'
          }}>
            Teachr uses cutting-edge AI to create a personalized learning experience. Our adaptive system analyzes your performance in real-time, adjusts question difficulty automatically, and provides instant help from our AI tutor. Track your progress, master skills at your own pace, and build confidence through intelligent practice.
          </p>
          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={onGetStarted}
              style={{
                padding: '20px 48px',
                fontSize: '18px',
                fontWeight: 600,
                background: 'transparent',
                color: '#00FFFF',
                border: '2px solid #00FFFF',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#00FFFF';
                e.currentTarget.style.color = '#0a0a0a';
                e.currentTarget.style.boxShadow = '0 0 40px rgba(0,255,255,0.6)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = '#00FFFF';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              Start Learning
            </button>
            <button
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              style={{
                padding: '20px 48px',
                fontSize: '18px',
                fontWeight: 600,
                background: 'transparent',
                color: 'rgba(255,255,255,0.7)',
                border: '2px solid rgba(255,255,255,0.3)',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.6)';
                e.currentTarget.style.color = '#FFFFFF';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.3)';
                e.currentTarget.style.color = 'rgba(255,255,255,0.7)';
              }}
            >
              Learn More
            </button>
          </div>
        </section>

        {/* Image Section */}
        <section style={{
          margin: '80px 0',
          textAlign: 'center'
        }}>
          <div style={{
            border: '1px solid rgba(0,255,255,0.2)',
            borderRadius: '16px',
            padding: '4px',
            background: 'linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,0,255,0.1) 100%)',
            display: 'inline-block',
            boxShadow: '0 0 60px rgba(0,255,255,0.2)',
            maxWidth: '1200px'
          }}>
            <div style={{
              background: '#0a0a0a',
              borderRadius: '12px',
              padding: '20px'
            }}>
              <img
                src="/landing-screenshots/home-screen-placeholder.png"
                alt="Teachr Interface"
                style={{
                  width: '100%',
                  height: 'auto',
                  borderRadius: '8px'
                }}
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                  target.parentElement!.innerHTML = `
                    <div style="
                      width: 100%;
                      height: 600px;
                      background: linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,0,255,0.1) 100%);
                      border-radius: 8px;
                      display: flex;
                      align-items: center;
                      justify-content: center;
                      margin: 0 auto;
                      color: #00FFFF;
                      font-size: 18px;
                      font-weight: 600;
                      border: 1px solid rgba(0,255,255,0.2);
                    ">
                      Learning Interface Preview
                    </div>
                  `;
                }}
              />
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" style={{
          padding: '120px 0'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '80px' }}>
            <h2 style={{
              fontSize: '56px',
              fontWeight: 800,
              marginBottom: '20px',
              background: 'linear-gradient(135deg, #00FFFF 0%, #FF00FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '-0.02em'
            }}>
              How It Works
            </h2>
            <p style={{
              fontSize: '20px',
              color: 'rgba(255,255,255,0.7)',
              maxWidth: '700px',
              margin: '0 auto'
            }}>
              A seamless learning experience designed for your success
            </p>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '40px',
            maxWidth: '1200px',
            margin: '0 auto'
          }}>
            {howItWorksSteps.map((step, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(0,255,255,0.05)',
                  border: '1px solid rgba(0,255,255,0.2)',
                  padding: '50px 40px',
                  borderRadius: '16px',
                  textAlign: 'center',
                  transition: 'all 0.3s ease',
                  position: 'relative',
                  overflow: 'hidden'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-8px)';
                  e.currentTarget.style.borderColor = 'rgba(0,255,255,0.5)';
                  e.currentTarget.style.boxShadow = '0 0 40px rgba(0,255,255,0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = 'rgba(0,255,255,0.2)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{
                  fontSize: '72px',
                  fontWeight: 900,
                  color: '#00FFFF',
                  marginBottom: '24px',
                  opacity: 0.9
                }}>
                  {step.step}
                </div>
                <h3 style={{
                  fontSize: '26px',
                  fontWeight: 700,
                  marginBottom: '20px',
                  color: '#00FFFF'
                }}>
                  {step.title}
                </h3>
                <p style={{
                  fontSize: '17px',
                  lineHeight: 1.7,
                  color: 'rgba(255,255,255,0.7)'
                }}>
                  {step.description}
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
              fontWeight: 800,
              marginBottom: '20px',
              background: 'linear-gradient(135deg, #00FFFF 0%, #FF00FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '-0.02em'
            }}>
              Powerful Features
            </h2>
            <p style={{
              fontSize: '20px',
              color: 'rgba(255,255,255,0.7)',
              maxWidth: '700px',
              margin: '0 auto'
            }}>
              Everything you need for effective, personalized learning
            </p>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '40px',
            maxWidth: '1200px',
            margin: '0 auto'
          }}>
            {landingFeatures.map((feature, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(0,255,255,0.05)',
                  border: '1px solid rgba(0,255,255,0.2)',
                  padding: '40px',
                  borderRadius: '16px',
                  textAlign: 'center',
                  transition: 'all 0.3s ease',
                  position: 'relative',
                  overflow: 'hidden'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-8px)';
                  e.currentTarget.style.borderColor = 'rgba(0,255,255,0.5)';
                  e.currentTarget.style.boxShadow = '0 0 40px rgba(0,255,255,0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = 'rgba(0,255,255,0.2)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{ fontSize: '56px', marginBottom: '24px' }}>{feature.icon}</div>
                <h3 style={{
                  fontSize: '24px',
                  fontWeight: 700,
                  marginBottom: '16px',
                  color: '#00FFFF'
                }}>
                  {feature.title}
                </h3>
                <p style={{
                  fontSize: '16px',
                  lineHeight: 1.7,
                  color: 'rgba(255,255,255,0.7)'
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
          background: 'linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,0,255,0.1) 100%)',
          border: '1px solid rgba(0,255,255,0.2)',
          borderRadius: '24px',
          margin: '80px 0',
          boxShadow: '0 0 60px rgba(0,255,255,0.1)'
        }}>
          <div style={{ maxWidth: '1000px', margin: '0 auto', textAlign: 'center' }}>
            <h2 style={{
              fontSize: '48px',
              fontWeight: 800,
              marginBottom: '40px',
              background: 'linear-gradient(135deg, #00FFFF 0%, #FF00FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            }}>
              Why Students Choose Teachr
            </h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '24px',
              textAlign: 'left'
            }}>
              {benefits.map((benefit, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '16px',
                    padding: '24px',
                    background: 'rgba(0,255,255,0.05)',
                    borderRadius: '16px',
                    border: '1px solid rgba(0,255,255,0.2)'
                  }}
                >
                  <div style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: '#00FFFF',
                    flexShrink: 0
                  }} />
                  <p style={{
                    fontSize: '17px',
                    fontWeight: 400,
                    margin: 0,
                    color: 'rgba(255,255,255,0.8)',
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
          padding: '120px 0',
          textAlign: 'center'
        }}>
          <h2 style={{
            fontSize: '64px',
            fontWeight: 800,
            marginBottom: '24px',
            background: 'linear-gradient(135deg, #00FFFF 0%, #FF00FF 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            letterSpacing: '-0.02em'
          }}>
            Ready to Excel?
          </h2>
          <p style={{
            fontSize: '22px',
            color: 'rgba(255,255,255,0.7)',
            marginBottom: '48px',
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
              background: 'transparent',
              color: '#00FFFF',
              border: '2px solid #00FFFF',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#00FFFF';
              e.currentTarget.style.color = '#0a0a0a';
              e.currentTarget.style.boxShadow = '0 0 50px rgba(0,255,255,0.7)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = '#00FFFF';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            Explore Learning
          </button>
          <p style={{
            fontSize: '15px',
            color: 'rgba(255,255,255,0.5)',
            marginTop: '32px'
          }}>
            Free to start • No credit card required • Cancel anytime
          </p>
        </section>

        {/* Footer */}
        <footer style={{
          padding: '60px 0',
          borderTop: '1px solid rgba(0,255,255,0.2)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '24px'
        }}>
          <p style={{ margin: 0, color: 'rgba(255,255,255,0.5)', fontSize: '14px' }}>
            © 2024 Teachr. Making learning accessible for everyone.
          </p>
          <div style={{ display: 'flex', gap: '32px' }}>
            <a href="#" style={{ color: 'rgba(255,255,255,0.5)', textDecoration: 'none', fontSize: '14px' }}>Privacy</a>
            <a href="#" style={{ color: 'rgba(255,255,255,0.5)', textDecoration: 'none', fontSize: '14px' }}>Terms</a>
            <a href="#" style={{ color: 'rgba(255,255,255,0.5)', textDecoration: 'none', fontSize: '14px' }}>Contact</a>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage5;
