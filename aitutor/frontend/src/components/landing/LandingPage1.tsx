/**
 * Landing Page 1: Neo-Brutalism Style
 * Matches the existing system design - Enhanced with more details
 */
import React from 'react';
import TeachrLogo from './TeachrLogo';
import './landing.scss';

interface LandingPage1Props {
  onGetStarted: () => void;
}

const LandingPage1: React.FC<LandingPage1Props> = ({ onGetStarted }) => {
  return (
    <div className="landing-page landing-page-1">
      <div className="landing-container">
        {/* Header */}
        <header className="landing-header">
          <TeachrLogo size="medium" />
          <nav style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
            <a href="#features" style={{ color: '#000000', textDecoration: 'none', fontWeight: 700, fontSize: '14px', textTransform: 'uppercase' }}>Features</a>
            <a href="#how-it-works" style={{ color: '#000000', textDecoration: 'none', fontWeight: 700, fontSize: '14px', textTransform: 'uppercase' }}>How It Works</a>
          </nav>
        </header>

        {/* Hero Section */}
        <section className="landing-hero">
          <div className="hero-content">
            <div style={{
              display: 'inline-block',
              padding: '8px 20px',
              background: '#FFD93D',
              border: '4px solid #000000',
              marginBottom: '24px',
              fontSize: '12px',
              fontWeight: 900,
              textTransform: 'uppercase',
              letterSpacing: '0.1em'
            }}>
              AI-Powered Learning Platform
            </div>
            <h1 className="hero-title">
              Learn Smarter, Not Harder
            </h1>
            <p className="hero-subtitle">
              Teachr adapts to your learning style, providing personalized practice questions that match your skill level. Get real-time help from our AI tutor, track your progress, and master any subject at your own pace.
            </p>
            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginTop: '32px' }}>
              <button
                onClick={onGetStarted}
                className="cta-button cta-button-neo"
              >
                Start Learning Free
              </button>
              <button
                onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
                style={{
                  padding: '16px 32px',
                  fontSize: '18px',
                  fontWeight: 700,
                  background: 'transparent',
                  color: '#000000',
                  border: '4px solid #000000',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  cursor: 'pointer',
                  transition: 'all 100ms ease-out'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#000000';
                  e.currentTarget.style.color = '#FFFFFF';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = '#000000';
                }}
              >
                Learn More
              </button>
            </div>
          </div>
          <div className="hero-image">
            <img
              src="/landing-screenshots/home-screen-placeholder.png"
              alt="Teachr Learning Interface"
              style={{ width: '100%', height: 'auto', borderRadius: '0' }}
            />
          </div>
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" style={{ padding: '100px 0', background: '#FFFFFF' }}>
          <h2 style={{
            fontSize: '56px',
            fontWeight: 900,
            textAlign: 'center',
            marginBottom: '16px',
            textTransform: 'uppercase',
            letterSpacing: '-0.02em'
          }}>
            How It Works
          </h2>
          <p style={{
            fontSize: '20px',
            fontWeight: 700,
            textAlign: 'center',
            marginBottom: '60px',
            color: '#000000'
          }}>
            Three simple steps to start your learning journey
          </p>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '40px',
            marginTop: '60px'
          }}>
            {[
              {
                number: '01',
                title: 'Sign Up & Set Your Grade',
                description: 'Create your account with Google and tell us your grade level. Our system automatically personalizes content for your age and learning stage.'
              },
              {
                number: '02',
                title: 'Practice with Adaptive Questions',
                description: 'Start practicing with questions that adapt to your skill level. Our AI analyzes your performance and adjusts difficulty in real-time.'
              },
              {
                number: '03',
                title: 'Get Help & Track Progress',
                description: 'Stuck on a problem? Ask our AI tutor for instant help. Watch your skills improve with detailed progress tracking for every topic.'
              }
            ].map((step, idx) => (
              <div
                key={idx}
                style={{
                  background: '#FFFFFF',
                  border: '4px solid #000000',
                  padding: '40px',
                  boxShadow: '8px 8px 0px 0px #000000',
                  position: 'relative'
                }}
              >
                <div style={{
                  fontSize: '72px',
                  fontWeight: 900,
                  color: '#FFD93D',
                  lineHeight: 1,
                  marginBottom: '20px',
                  WebkitTextStroke: '4px #000000',

                  // For future standard compliance or if browsers drop prefix:
                  // textStroke: '4px #000000' 
                  // But 'textStroke' is not in standard React CSSProperties yet.
                }}>
                  {step.number}
                </div>
                <h3 style={{
                  fontSize: '24px',
                  WebkitTextStroke: "2px black",
                  fontWeight: 900,
                  marginBottom: '16px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}>
                  {step.title}
                </h3>
                <p style={{
                  fontSize: '16px',
                  fontWeight: 700,
                  lineHeight: 1.6,
                  color: '#000000'
                }}>
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="landing-features">
          <div style={{ textAlign: 'center', marginBottom: '60px' }}>
            <h2 style={{
              fontSize: '56px',
              fontWeight: 900,
              marginBottom: '16px',
              textTransform: 'uppercase',
              letterSpacing: '-0.02em'
            }}>
              Powerful Features
            </h2>
            <p style={{
              fontSize: '20px',
              fontWeight: 700,
              color: '#000000'
            }}>
              Everything you need to succeed in your learning journey
            </p>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '32px'
          }}>
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>Adaptive Learning System</h3>
              <p>Our intelligent system analyzes your performance and automatically adjusts question difficulty. Questions get harder as you improve and easier when you need more practice. No two students see the same learning path.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🤖</div>
              <h3>AI-Powered Real-Time Tutoring</h3>
              <p>Get instant help from our AI tutor powered by Google Gemini. Ask questions via voice or text, receive step-by-step explanations, and get personalized guidance whenever you're stuck. Available 24/7.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>Comprehensive Progress Tracking</h3>
              <p>Track your improvement with detailed analytics. See memory strength for each skill, practice counts, accuracy rates, and identify areas that need more attention. Visualize your learning journey.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">✨</div>
              <h3>Interactive Question Types</h3>
              <p>Engage with diverse question formats including multiple choice, numeric input, graphing exercises, drag-and-drop, and more. Rich, interactive content makes learning engaging and effective.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎓</div>
              <h3>Grade-Appropriate Content</h3>
              <p>Content automatically matches your grade level from Kindergarten through 12th grade. The system ensures you're always practicing at the right level for your age and learning stage.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Instant Feedback & Corrections</h3>
              <p>Get immediate feedback on every answer. Learn from mistakes with detailed explanations and understand concepts better through real-time corrections and hints.</p>
            </div>
          </div>
        </section>

        {/* Benefits Section */}
        <section style={{
          padding: '100px 0',
          background: '#FFD93D',
          border: '4px solid #000000',
          margin: '60px 0',
          boxShadow: '12px 12px 0px 0px #000000'
        }}>
          <h2 style={{
            fontSize: '48px',
            fontWeight: 900,
            textAlign: 'center',
            marginBottom: '40px',
            textTransform: 'uppercase',
            letterSpacing: '-0.02em'
          }}>
            Why Students Love Teachr
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '32px',
            maxWidth: '1000px',
            margin: '0 auto'
          }}>
            {[
              'Practice at your own pace - no pressure, no deadlines',
              'Questions adapt to your skill level automatically',
              'Get help instantly from AI tutor when stuck',
              'Track progress and see improvement over time',
              'Learn from mistakes with detailed explanations',
              'Engaging interactive questions keep you motivated'
            ].map((benefit, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                  padding: '20px',
                  background: '#FFFFFF',
                  border: '4px solid #000000',
                  boxShadow: '4px 4px 0px 0px #000000'
                }}
              >
                <div style={{
                  width: '24px',
                  height: '24px',
                  background: '#000000',
                  flexShrink: 0
                }} />
                <p style={{
                  fontSize: '18px',
                  fontWeight: 700,
                  margin: 0,
                  color: '#000000'
                }}>
                  {benefit}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="landing-cta">
          <h2>Ready to Transform Your Learning?</h2>
          <p>Join thousands of students who are mastering new skills and building confidence every day with Teachr.</p>
          <button
            onClick={onGetStarted}
            className="cta-button cta-button-neo cta-button-large"
          >
            Get Started Now
          </button>
          <p style={{
            fontSize: '14px',
            fontWeight: 700,
            marginTop: '24px',
            color: '#000000',
            opacity: 0.7
          }}>
            Free to start • No credit card required • Cancel anytime
          </p>
        </section>

        {/* Footer */}
        <footer className="landing-footer">
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '24px'
          }}>
            <p style={{ margin: 0 }}>© 2024 Teachr. Making learning accessible for everyone.</p>
            <div style={{ display: 'flex', gap: '24px' }}>
              <a href="#" style={{ color: '#000000', textDecoration: 'none', fontWeight: 700, fontSize: '14px' }}>Privacy</a>
              <a href="#" style={{ color: '#000000', textDecoration: 'none', fontWeight: 700, fontSize: '14px' }}>Terms</a>
              <a href="#" style={{ color: '#000000', textDecoration: 'none', fontWeight: 700, fontSize: '14px' }}>Contact</a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage1;
