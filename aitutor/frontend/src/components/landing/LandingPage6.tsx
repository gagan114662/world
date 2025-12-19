/**
 * Landing Page 6: Card-Based Layout
 * Feature cards, grid system
 */
import React from 'react';
import TeachrLogo from './TeachrLogo';
import { landingFeatures, howItWorksSteps, benefits } from './landingData';

interface LandingPage6Props {
  onGetStarted: () => void;
}

const LandingPage6: React.FC<LandingPage6Props> = ({ onGetStarted }) => {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#f8f9fa',
      color: '#212529',
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
          <TeachrLogo size="medium" color="#212529" />
          <nav style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
            <a href="#features" style={{ color: '#666666', textDecoration: 'none', fontSize: '15px', fontWeight: 500 }}>Features</a>
            <button
              onClick={onGetStarted}
              style={{
                padding: '12px 28px',
                fontSize: '15px',
                fontWeight: 600,
                background: '#007bff',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#0056b3';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#007bff';
              }}
            >
              Get Started
            </button>
          </nav>
        </header>

        {/* Hero Section */}
        <section style={{
          padding: '80px 0',
          textAlign: 'center'
        }}>
          <div style={{
            display: 'inline-block',
            padding: '6px 16px',
            background: '#e9ecef',
            borderRadius: '20px',
            fontSize: '13px',
            fontWeight: 600,
            color: '#495057',
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
            color: '#212529',
            letterSpacing: '-0.03em'
          }}>
            Build Skills That Last
          </h1>
          <p style={{
            fontSize: '22px',
            fontWeight: 400,
            lineHeight: 1.7,
            marginBottom: '48px',
            color: '#6c757d',
            maxWidth: '800px',
            margin: '0 auto 48px'
          }}>
            Teachr uses advanced AI to create a personalized learning experience. Our adaptive system analyzes your performance in real-time, adjusts question difficulty automatically, and provides instant help from our AI tutor. Track your progress, master skills at your own pace, and build confidence through intelligent practice.
          </p>
          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={onGetStarted}
              style={{
                padding: '18px 40px',
                fontSize: '18px',
                fontWeight: 600,
                background: '#007bff',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                boxShadow: '0 4px 12px rgba(0,123,255,0.3)',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#0056b3';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,123,255,0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#007bff';
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,123,255,0.3)';
              }}
            >
              Start Practicing
            </button>
            <button
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              style={{
                padding: '18px 40px',
                fontSize: '18px',
                fontWeight: 600,
                background: 'transparent',
                color: '#007bff',
                border: '2px solid #007bff',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#f0f7ff';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
              }}
            >
              Learn More
            </button>
          </div>
        </section>

        {/* Image Card */}
        <section style={{ margin: '60px 0' }}>
          <div style={{
            background: '#FFFFFF',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
            overflow: 'hidden'
          }}>
            <img
              src="/landing-screenshots/home-screen-placeholder.png"
              alt="Teachr Interface"
              style={{
                width: '100%',
                height: 'auto',
                borderRadius: '12px'
              }}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.parentElement!.innerHTML = `
                  <div style="
                    width: 100%;
                    height: 500px;
                    background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #6c757d;
                    font-size: 18px;
                    font-weight: 500;
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
              color: '#212529',
              letterSpacing: '-0.02em'
            }}>
              How It Works
            </h2>
            <p style={{
              fontSize: '20px',
              color: '#6c757d',
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
            {howItWorksSteps.map((step, idx) => (
              <div key={idx}>
                <div style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: '#007bff',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px',
                  fontWeight: 700,
                  marginBottom: '32px'
                }}>
                  {step.step}
                </div>
                <h3 style={{
                  fontSize: '26px',
                  fontWeight: 600,
                  marginBottom: '16px',
                  color: '#212529'
                }}>
                  {step.title}
                </h3>
                <p style={{
                  fontSize: '17px',
                  lineHeight: 1.7,
                  color: '#6c757d'
                }}>
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Feature Cards Grid */}
        <section id="features" style={{
          padding: '120px 0'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '80px' }}>
            <h2 style={{
              fontSize: '56px',
              fontWeight: 700,
              marginBottom: '20px',
              color: '#212529',
              letterSpacing: '-0.02em'
            }}>
              Powerful Features
            </h2>
            <p style={{
              fontSize: '20px',
              color: '#6c757d',
              maxWidth: '600px',
              margin: '0 auto'
            }}>
              Everything you need for effective, personalized learning
            </p>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '32px',
            maxWidth: '1200px',
            margin: '0 auto'
          }}>
            {landingFeatures.slice(0, 6).map((feature, idx) => {
              const colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#20c997'];
              return (
                <div
                  key={idx}
                  style={{
                    background: '#FFFFFF',
                    borderRadius: '16px',
                    padding: '40px 32px',
                    boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
                    transition: 'all 0.3s ease',
                    borderTop: `4px solid ${colors[idx]}`
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-8px)';
                    e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.12)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 4px 24px rgba(0,0,0,0.08)';
                  }}
                >
                  <div style={{ fontSize: '48px', marginBottom: '20px' }}>{feature.icon}</div>
                  <h3 style={{
                    fontSize: '22px',
                    fontWeight: 700,
                    marginBottom: '16px',
                    color: '#212529'
                  }}>
                    {feature.title}
                  </h3>
                  <p style={{
                    fontSize: '16px',
                    lineHeight: 1.7,
                    color: '#6c757d'
                  }}>
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </section>

        {/* Benefits Section */}
        <section style={{
          padding: '100px 0',
          background: '#f8f9fa',
          borderRadius: '16px',
          margin: '60px 0'
        }}>
          <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <h2 style={{
              fontSize: '48px',
              fontWeight: 700,
              textAlign: 'center',
              marginBottom: '40px',
              color: '#212529',
              letterSpacing: '-0.02em'
            }}>
              Why Students Choose Teachr
            </h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '24px'
            }}>
              {benefits.map((benefit, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '16px',
                    padding: '24px',
                    background: '#FFFFFF',
                    borderRadius: '12px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                  }}
                >
                  <div style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: '#007bff',
                    flexShrink: 0,
                    marginTop: '2px'
                  }} />
                  <p style={{
                    fontSize: '17px',
                    fontWeight: 400,
                    margin: 0,
                    color: '#495057',
                    lineHeight: 1.6
                  }}>
                    {benefit}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Card */}
        <section style={{
          padding: '80px 0'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #007bff 0%, #0056b3 100%)',
            borderRadius: '24px',
            padding: '80px 40px',
            textAlign: 'center',
            color: '#FFFFFF',
            boxShadow: '0 8px 32px rgba(0,123,255,0.3)'
          }}>
            <h2 style={{
              fontSize: '48px',
              fontWeight: 700,
              marginBottom: '20px',
              color: '#FFFFFF'
            }}>
              Ready to Transform Your Learning?
            </h2>
          <p style={{
            fontSize: '20px',
            marginBottom: '40px',
            opacity: 0.9
          }}>
            Join thousands of students who are mastering new skills and building confidence every day.
          </p>
            <button
              onClick={onGetStarted}
              style={{
                padding: '18px 48px',
                fontSize: '18px',
                fontWeight: 600,
                background: '#FFFFFF',
                color: '#007bff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-3px) scale(1.05)';
                e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,0,0,0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
              }}
            >
              Get Started Now
            </button>
          </div>
        </section>

        {/* Footer */}
        <footer style={{
          padding: '60px 0',
          borderTop: '1px solid #dee2e6',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '24px'
        }}>
          <p style={{ margin: 0, color: '#6c757d', fontSize: '14px' }}>
            © 2024 Teachr. Making learning accessible for everyone.
          </p>
          <div style={{ display: 'flex', gap: '32px' }}>
            <a href="#" style={{ color: '#6c757d', textDecoration: 'none', fontSize: '14px' }}>Privacy</a>
            <a href="#" style={{ color: '#6c757d', textDecoration: 'none', fontSize: '14px' }}>Terms</a>
            <a href="#" style={{ color: '#6c757d', textDecoration: 'none', fontSize: '14px' }}>Contact</a>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage6;

