/**
 * Landing Page 3: Gradient Modern Design - Enhanced
 * Vibrant gradients, smooth animations, detailed content
 */
import React, { useEffect, useState } from 'react';
import TeachrLogo from './TeachrLogo';

interface LandingPage3Props {
  onGetStarted: () => void;
}

const LandingPage3: React.FC<LandingPage3Props> = ({ onGetStarted }) => {
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
      {/* Animated background elements */}
      <div style={{
        position: 'absolute',
        top: '-20%',
        right: '-10%',
        width: '600px',
        height: '600px',
        background: 'rgba(255,255,255,0.1)',
        borderRadius: '50%',
        filter: 'blur(100px)',
        animation: 'float 20s ease-in-out infinite'
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-20%',
        left: '-10%',
        width: '500px',
        height: '500px',
        background: 'rgba(255,255,255,0.1)',
        borderRadius: '50%',
        filter: 'blur(100px)',
        animation: 'float 25s ease-in-out infinite reverse'
      }} />

      <style>{`
        @keyframes float {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(40px, -40px) scale(1.1); }
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

      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 40px', position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <header style={{
          padding: '40px 0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          opacity: isVisible ? 1 : 0,
          animation: isVisible ? 'fadeInUp 0.6s ease-out' : 'none'
        }}>
          <TeachrLogo size="medium" color="#FFFFFF" />
          <nav style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
            <a href="#features" style={{ color: 'rgba(255,255,255,0.9)', textDecoration: 'none', fontSize: '15px', fontWeight: 500 }}>Features</a>
            <button
              onClick={onGetStarted}
              style={{
                padding: '12px 28px',
                fontSize: '15px',
                fontWeight: 600,
                background: 'rgba(255,255,255,0.2)',
                color: '#FFFFFF',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                backdropFilter: 'blur(10px)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.2)';
              }}
            >
              Get Started
            </button>
          </nav>
        </header>

        {/* Hero Section */}
        <section style={{
          padding: '120px 0 100px',
          textAlign: 'center'
        }}>
          <div style={{
            opacity: isVisible ? 1 : 0,
            animation: isVisible ? 'fadeInUp 0.8s ease-out 0.2s both' : 'none'
          }}>
            <div style={{
              display: 'inline-block',
              padding: '8px 20px',
              background: 'rgba(255,255,255,0.2)',
              backdropFilter: 'blur(10px)',
              borderRadius: '30px',
              fontSize: '13px',
              fontWeight: 600,
              marginBottom: '32px',
              border: '1px solid rgba(255,255,255,0.3)'
            }}>
              AI-Powered Adaptive Learning
            </div>
            <h1 style={{
              fontSize: '80px',
              fontWeight: 800,
              lineHeight: 1.1,
              marginBottom: '32px',
              background: 'linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
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
              color: 'rgba(255,255,255,0.95)',
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
                Try It Free
              </button>
              <button
                onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
                style={{
                  padding: '20px 48px',
                  fontSize: '18px',
                  fontWeight: 600,
                  background: 'transparent',
                  color: '#FFFFFF',
                  border: '2px solid rgba(255,255,255,0.3)',
                  borderRadius: '50px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  backdropFilter: 'blur(10px)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.5)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.3)';
                }}
              >
                Learn More
              </button>
            </div>
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
            background: 'rgba(255,255,255,0.15)',
            backdropFilter: 'blur(20px)',
            borderRadius: '32px',
            padding: '32px',
            display: 'inline-block',
            border: '1px solid rgba(255,255,255,0.2)',
            boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
            maxWidth: '1200px'
          }}>
            <img
              src="/landing-screenshots/home-screen-placeholder.png"
              alt="Teachr Interface"
              style={{
                width: '100%',
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
                    height: 600px;
                    background: rgba(255,255,255,0.2);
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

        {/* How It Works Section */}
        <section id="how-it-works" style={{
          padding: '120px 0',
          opacity: isVisible ? 1 : 0,
          animation: isVisible ? 'fadeInUp 1s ease-out 0.6s both' : 'none'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '80px' }}>
            <h2 style={{
              fontSize: '56px',
              fontWeight: 800,
              marginBottom: '20px',
              color: '#FFFFFF',
              letterSpacing: '-0.02em'
            }}>
              How It Works
            </h2>
            <p style={{
              fontSize: '20px',
              color: 'rgba(255,255,255,0.9)',
              maxWidth: '600px',
              margin: '0 auto'
            }}>
              A seamless learning experience in three simple steps
            </p>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '40px',
            maxWidth: '1200px',
            margin: '0 auto'
          }}>
            {[
              {
                number: '01',
                title: 'Sign Up & Personalize',
                description: 'Create your account with Google in seconds. Set your grade level, and our AI system automatically personalizes content, question difficulty, and learning paths tailored to your age and skill level.'
              },
              {
                number: '02',
                title: 'Practice with AI Adaptation',
                description: 'Start practicing with questions that adapt in real-time. Our advanced AI analyzes every answer, tracks your performance patterns, and automatically adjusts difficulty to keep you challenged at the optimal level.'
              },
              {
                number: '03',
                title: 'Get Help & Track Growth',
                description: 'Stuck on a problem? Our AI tutor provides instant explanations via voice or text. Monitor your progress with comprehensive analytics showing memory strength, accuracy, and skill mastery over time.'
              }
            ].map((step, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(255,255,255,0.15)',
                  backdropFilter: 'blur(20px)',
                  padding: '50px 40px',
                  borderRadius: '24px',
                  textAlign: 'center',
                  border: '1px solid rgba(255,255,255,0.2)',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-10px)';
                  e.currentTarget.style.background = 'rgba(255,255,255,0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.background = 'rgba(255,255,255,0.15)';
                }}
              >
                <div style={{
                  fontSize: '64px',
                  fontWeight: 900,
                  color: '#FFFFFF',
                  marginBottom: '24px',
                  opacity: 0.9
                }}>
                  {step.number}
                </div>
                <h3 style={{
                  fontSize: '26px',
                  fontWeight: 700,
                  marginBottom: '20px',
                  color: '#FFFFFF'
                }}>
                  {step.title}
                </h3>
                <p style={{
                  fontSize: '17px',
                  lineHeight: 1.7,
                  color: 'rgba(255,255,255,0.9)'
                }}>
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Features Section */}
        <section id="features" style={{
          padding: '120px 0',
          opacity: isVisible ? 1 : 0,
          animation: isVisible ? 'fadeInUp 1s ease-out 0.8s both' : 'none'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '80px' }}>
            <h2 style={{
              fontSize: '56px',
              fontWeight: 800,
              marginBottom: '20px',
              color: '#FFFFFF',
              letterSpacing: '-0.02em'
            }}>
              Powerful Features
            </h2>
            <p style={{
              fontSize: '20px',
              color: 'rgba(255,255,255,0.9)',
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
            {[
              {
                icon: '🎯',
                title: 'Intelligent Adaptive Learning',
                description: 'Our advanced algorithm uses performance data to create a unique learning path for each student. Questions automatically adjust in difficulty based on your answers, ensuring optimal challenge. The system learns your learning patterns, strengths, and areas for improvement.'
              },
              {
                icon: '🤖',
                title: 'AI Tutor with Context Awareness',
                description: 'Powered by Google Gemini Live API, our AI tutor provides real-time assistance via voice or text. It understands context, remembers your learning history, and provides explanations tailored to your level. Get help 24/7 with step-by-step guidance.'
              },
              {
                icon: '📊',
                title: 'Advanced Progress Analytics',
                description: 'Track your improvement with comprehensive metrics including memory strength per skill, practice frequency, accuracy trends, and time analysis. Visualize your learning journey with detailed charts and identify skills that need more attention.'
              },
              {
                icon: '✨',
                title: 'Rich Interactive Content',
                description: 'Engage with diverse question formats including multiple choice, numeric input, graphing exercises, drag-and-drop matching, expression input, and more. Each format tests different skills and keeps learning engaging and effective.'
              },
              {
                icon: '🎓',
                title: 'Grade-Level Personalization',
                description: 'Content automatically matches your grade level from K-12. The system ensures appropriate difficulty, smooth progression, and age-appropriate content. As you advance, content naturally becomes more challenging.'
              },
              {
                icon: '⚡',
                title: 'Instant Feedback & Learning',
                description: 'Receive immediate feedback on every answer with detailed explanations. Learn from mistakes in real-time, understand concepts deeply through corrections, and access contextual hints. Feedback is personalized to help you understand not just what was wrong, but why.'
              }
            ].map((feature, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(255,255,255,0.1)',
                  backdropFilter: 'blur(15px)',
                  padding: '40px',
                  borderRadius: '24px',
                  textAlign: 'center',
                  border: '1px solid rgba(255,255,255,0.2)',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-8px)';
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
                  lineHeight: 1.7,
                  color: 'rgba(255,255,255,0.9)'
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
          background: 'rgba(255,255,255,0.1)',
          backdropFilter: 'blur(20px)',
          borderRadius: '32px',
          margin: '80px 0',
          border: '1px solid rgba(255,255,255,0.2)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.2)'
        }}>
          <div style={{ maxWidth: '1000px', margin: '0 auto', textAlign: 'center' }}>
            <h2 style={{
              fontSize: '48px',
              fontWeight: 800,
              marginBottom: '40px',
              color: '#FFFFFF',
              letterSpacing: '-0.02em'
            }}>
              Why Students Choose Teachr
            </h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '24px',
              textAlign: 'left'
            }}>
              {[
                'Practice at your own pace with no pressure or deadlines',
                'Questions adapt automatically to your skill level',
                'Get instant help from AI tutor available 24/7',
                'Track detailed progress and see improvement over time',
                'Learn from mistakes with comprehensive explanations',
                'Engaging interactive content keeps you motivated'
              ].map((benefit, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '16px',
                    padding: '24px',
                    background: 'rgba(255,255,255,0.1)',
                    borderRadius: '16px',
                    border: '1px solid rgba(255,255,255,0.2)'
                  }}
                >
                  <div style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: '#FFFFFF',
                    flexShrink: 0
                  }} />
                  <p style={{
                    fontSize: '17px',
                    fontWeight: 400,
                    margin: 0,
                    color: 'rgba(255,255,255,0.95)',
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
          textAlign: 'center',
          opacity: isVisible ? 1 : 0,
          animation: isVisible ? 'fadeInUp 1s ease-out 1s both' : 'none'
        }}>
          <h2 style={{
            fontSize: '64px',
            fontWeight: 800,
            marginBottom: '24px',
            background: 'linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            letterSpacing: '-0.02em'
          }}>
            Ready to Excel?
          </h2>
          <p style={{
            fontSize: '22px',
            color: 'rgba(255,255,255,0.9)',
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
          <p style={{
            fontSize: '15px',
            color: 'rgba(255,255,255,0.7)',
            marginTop: '32px'
          }}>
            Free to start • No credit card required • Cancel anytime
          </p>
        </section>

        {/* Footer */}
        <footer style={{
          padding: '60px 0',
          borderTop: '1px solid rgba(255,255,255,0.2)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '24px'
        }}>
          <p style={{ margin: 0, color: 'rgba(255,255,255,0.7)', fontSize: '14px' }}>
            © 2024 Teachr. Making learning accessible for everyone.
          </p>
          <div style={{ display: 'flex', gap: '32px' }}>
            <a href="#" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', fontSize: '14px' }}>Privacy</a>
            <a href="#" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', fontSize: '14px' }}>Terms</a>
            <a href="#" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', fontSize: '14px' }}>Contact</a>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage3;
