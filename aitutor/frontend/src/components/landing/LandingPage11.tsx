/**
 * Landing Page 11: "Understanding" Angle - Bob Hoffman Style
 * Neo-Brutalism with edgy, direct copy
 */
import React from 'react';
import './landing.scss';

interface LandingPage11Props {
  onGetStarted: () => void;
}

const LandingPage11: React.FC<LandingPage11Props> = ({ onGetStarted }) => {
  return (
    <div className="landing-page landing-page-11">
      {/* Noise texture overlay */}
      <div className="noise-overlay" />

      <div className="landing-container">
        {/* Header */}
        <header className="landing-header">
          <img src="/logo.png" alt="Teachr" className="header-logo" />
          <button
            onClick={onGetStarted}
            className="header-cta"
          >
            Try It Free
          </button>
        </header>

        {/* Hero Section */}
        <section className="hero-section">
          <div className="hero-badge">
            <span>MATH TUTOR</span>
            <span className="badge-dot" />
            <span>24/7</span>
          </div>

          <h1 className="hero-headline">
            <span className="headline-line">What If You</span>
            <span className="headline-line headline-accent">Actually</span>
            <span className="headline-line">Understood It?</span>
          </h1>

          <p className="hero-subhead">
            Not memorized it for the test. Not faked your way through it.
            <strong> Actually got it.</strong> That's what we're building.
          </p>

          <div className="hero-cta-group">
            <button onClick={onGetStarted} className="cta-primary">
              Start Learning Now
            </button>
            <div className="cta-note">Free to start. No credit card.</div>
          </div>

          {/* Decorative elements */}
          <div className="hero-decoration">
            <div className="deco-star deco-star-1">★</div>
            <div className="deco-star deco-star-2">★</div>
            <div className="deco-block deco-block-1" />
            <div className="deco-block deco-block-2" />
          </div>
        </section>

        {/* Problem Section */}
        <section className="problem-section">
          <div className="section-label">THE REAL PROBLEM</div>
          <h2 className="section-title">
            You Don't Need More Content.<br />
            <span className="title-accent">You Need Better Understanding.</span>
          </h2>

          <div className="problem-content">
            <p className="problem-text">
              The internet has more educational content than any human could consume in ten lifetimes.
            </p>
            <p className="problem-text">
              Khan Academy. YouTube. Coursera. Udemy. Brilliant. That random guy with a whiteboard who pronounces "integral" weird.
            </p>
            <div className="callout-box">
              <p>
                You're not struggling because there's not enough out there.
                You're struggling because <strong>none of it was built for you.</strong>
              </p>
            </div>
            <p className="problem-text">
              Generic explanations. One-size-fits-all pacing. Videos that assume you understood minute three when you got lost at minute one.
            </p>
            <p className="problem-conclusion">
              More content isn't the answer. <strong>Better teaching is.</strong>
            </p>
          </div>
        </section>

        {/* Teaching vs Content Section */}
        <section className="difference-section">
          <div className="section-label">THE DIFFERENCE</div>
          <h2 className="section-title">Teaching vs. Content</h2>

          <div className="comparison-grid">
            <div className="comparison-card comparison-bad">
              <div className="comparison-label">CONTENT</div>
              <ul className="comparison-list">
                <li>A video you pause, rewind, pause again, then give up on.</li>
                <li>Moves at its own pace.</li>
                <li>Doesn't know if you understood.</li>
              </ul>
            </div>
            <div className="comparison-card comparison-good">
              <div className="comparison-label">TEACHING</div>
              <ul className="comparison-list">
                <li>Someone who notices you're confused and tries a different angle.</li>
                <li>Moves at <em>yours</em>.</li>
                <li>Checks. And adjusts. And checks again.</li>
              </ul>
            </div>
          </div>

          <div className="difference-conclusion">
            <p>teachr isn't content. <strong>It's teaching.</strong></p>
          </div>
        </section>

        {/* What Changes Section */}
        <section className="changes-section">
          <div className="section-label">WHAT ACTUALLY CHANGES</div>
          <h2 className="section-title">When Learning Finally Works</h2>

          <div className="changes-grid">
            {[
              {
                title: "You stop avoiding the subject.",
                desc: "That class you dread? The homework you leave until last? It's not because you're bad at it. It's because no one's taught you properly yet."
              },
              {
                title: "You build real confidence.",
                desc: "Not the fake kind where you hope the test questions are easy. The kind where you walk in knowing you've got this."
              },
              {
                title: "You save time.",
                desc: "No more watching 45-minute videos for 3 minutes of relevance. No more hunting through forums. Just direct answers."
              },
              {
                title: "You start to like it.",
                desc: "Controversial opinion: most subjects are interesting when you actually understand them. Math isn't boring. Bad teaching is boring."
              }
            ].map((item, idx) => (
              <div key={idx} className="change-card">
                <div className="change-number">{String(idx + 1).padStart(2, '0')}</div>
                <h3 className="change-title">{item.title}</h3>
                <p className="change-desc">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* How It Works Section */}
        <section className="how-section">
          <div className="section-label">HOW IT WORKS</div>
          <h2 className="section-title">Simple. Like It Should Be.</h2>

          <div className="steps-grid">
            {[
              {
                num: "1",
                title: "Bring Your Problem",
                desc: "A concept. A homework question. A \"why does this even exist\" rant. All valid starting points."
              },
              {
                num: "2",
                title: "Learn Your Way",
                desc: "Visual learner? We'll draw it out. Need real-world examples? Got those. Want the formal proof? Sure, weirdo, we have that too."
              },
              {
                num: "3",
                title: "Move On When You're Ready",
                desc: "Not when the video ends. Not when the bell rings. When you actually get it."
              }
            ].map((step, idx) => (
              <div key={idx} className="step-card">
                <div className="step-number">{step.num}</div>
                <h3 className="step-title">{step.title}</h3>
                <p className="step-desc">{step.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Subjects Section */}
        <section className="subjects-section">
          <div className="subjects-content">
            <div className="section-label">CURRENTLY TEACHING</div>
            <h2 className="section-title">Math. All of It.</h2>

            <div className="subjects-list">
              {['Arithmetic', 'Algebra', 'Geometry', 'Trigonometry', 'Calculus', 'Statistics'].map((subject, idx) => (
                <span key={idx} className="subject-tag">{subject}</span>
              ))}
            </div>

            <div className="coming-soon">
              <span className="coming-badge">COMING SOON</span>
              <span>Chemistry & Biology</span>
            </div>
          </div>
        </section>

        {/* Honest Truth Section */}
        <section className="honest-section">
          <div className="honest-card">
            <div className="section-label">THE HONEST TRUTH</div>
            <h2 className="honest-title">This Works If You Work</h2>
            <p className="honest-text">
              We're not selling magic. We're selling a better tool.
            </p>
            <p className="honest-text">
              You still have to show up. You still have to try. You still have to do the practice problems.
            </p>
            <p className="honest-text">
              But when you do? <strong>You'll actually understand them.</strong> Not just survive them.
            </p>
            <div className="honest-conclusion">
              The difference between "I passed" and "I get it" is better teaching.<br />
              <strong>That's what we're here for.</strong>
            </div>
          </div>
        </section>

        {/* Who Is This For Section */}
        <section className="audience-section">
          <div className="section-label">BE HONEST. IS THIS YOU?</div>
          <h2 className="section-title">Who Gets the Most Out of This</h2>

          <div className="audience-list">
            {[
              "You've told yourself you're \"just not a math person\"",
              "You've watched the same concept explained 10 different ways and still don't get it",
              "You're tired of feeling behind",
              "You learn differently and regular classes don't work for you",
              "You want to actually understand, not just pass"
            ].map((item, idx) => (
              <div key={idx} className="audience-item">
                <div className="check-box">✓</div>
                <span>{item}</span>
              </div>
            ))}
          </div>

          <p className="audience-conclusion">
            If you nodded at any of those, <strong>we built this for you.</strong>
          </p>
        </section>

        {/* What We're Not Section */}
        <section className="not-section">
          <div className="section-label">CLARITY IS KINDNESS</div>
          <h2 className="section-title">What We're Not</h2>

          <div className="not-grid">
            {[
              {
                title: "We're not a homework cheat code.",
                desc: "If you want answers without understanding, go somewhere else. We're not interested."
              },
              {
                title: "We're not replacing your teachers.",
                desc: "Good teachers are irreplaceable. We're the support system for when they're not available."
              },
              {
                title: "We're not a miracle.",
                desc: "We're a tool. A really good one. But you still have to use it."
              }
            ].map((item, idx) => (
              <div key={idx} className="not-card">
                <h3>{item.title}</h3>
                <p>{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Final CTA */}
        <section className="final-cta">
          <div className="cta-content">
            <div className="section-label">THE PITCH</div>
            <h2 className="cta-headline">One Sentence. No Tricks.</h2>
            <p className="cta-pitch">
              A personalized math tutor that's available 24/7, adapts to how you learn, and actually helps you understand.
            </p>
            <p className="cta-simple">That's it. That's the product.</p>
            <button onClick={onGetStarted} className="cta-primary cta-large">
              Start Learning Now
            </button>
          </div>
        </section>

        {/* Footer */}
        <footer className="landing-footer-11">
          <div className="footer-content">
            <img src="/logo.png" alt="Teachr" className="footer-logo" />
            <p className="footer-tagline">Finally, teaching that teaches.</p>
            <p className="footer-motto">Because understanding beats memorizing. Every time.</p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage11;
