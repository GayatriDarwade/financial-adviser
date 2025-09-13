import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
import time

# Page configuration
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .analysis-section {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def create_agent():
    """Create the financial analysis agent"""
    try:
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[
                ReasoningTools(),
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    company_info=True,
                    company_news=True
                ),
            ],
            instructions="""
            You are an expert financial analysis agent. When providing information:
            1. Use clear, structured tables to display data
            2. Always think through your analysis step by step using reasoning
            3. Provide detailed reasoning for your recommendations
            4. Format numerical data appropriately (currency, percentages)
            5. Highlight key insights and risks
            6. Be objective and data-driven in your analysis
            7. Explain complex financial concepts in accessible terms
            """,
            markdown=True,
            show_tool_calls=True,
        )
        return agent
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        return None

def get_stock_data(symbols, period="6mo"):
    """Get basic stock data for visualization"""
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            data[symbol] = {
                'history': hist,
                'info': info,
                'current_price': hist['Close'].iloc[-1] if not hist.empty else None
            }
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")
    return data

def create_price_chart(stock_data):
    """Create an interactive price chart"""
    fig = go.Figure()
    
    for symbol, data in stock_data.items():
        if not data['history'].empty:
            fig.add_trace(go.Scatter(
                x=data['history'].index,
                y=data['history']['Close'],
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Stock Price Comparison (6 Months)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )
    
    return fig

def create_volume_chart(stock_data):
    """Create volume comparison chart"""
    fig = go.Figure()
    
    for symbol, data in stock_data.items():
        if not data['history'].empty:
            fig.add_trace(go.Bar(
                x=data['history'].index[-30:],  # Last 30 days
                y=data['history']['Volume'][-30:],
                name=symbol,
                opacity=0.7
            ))
    
    fig.update_layout(
        title="Trading Volume Comparison (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_white",
        height=400
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">AI Financial Advisor</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Google Gemini & Advanced Financial Analysis Tools**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Agent initialization
        if st.button("🚀 Initialize AI Agent", type="primary"):
            with st.spinner("Initializing AI Agent..."):
                st.session_state.agent = create_agent()
                if st.session_state.agent:
                    st.success("✅ Agent initialized successfully!")
                else:
                    st.error("❌ Failed to initialize agent")
        
        st.divider()
        
        # Stock selection
        st.header("📊 Stock Selection")
        default_stocks = ["AAPL", "TSLA", "GOOGL"]
        
        # Predefined stock options
        popular_stocks = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
            "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"]
        }
        
        selected_category = st.selectbox("Choose Category:", list(popular_stocks.keys()))
        
        # Multi-select for stocks
        selected_stocks = st.multiselect(
            "Select Stocks:",
            popular_stocks[selected_category],
            default=default_stocks if selected_category == "Technology" else popular_stocks[selected_category][:3]
        )
        
        # Custom stock input
        custom_stocks = st.text_input(
            "Or enter custom symbols (comma-separated):",
            placeholder="e.g., AAPL, MSFT, AMZN"
        )
        
        if custom_stocks:
            custom_list = [s.strip().upper() for s in custom_stocks.split(",")]
            selected_stocks = custom_list
        
        if not selected_stocks:
            selected_stocks = default_stocks
        
        st.divider()
        
        # Analysis options
        st.header("📋 Analysis Options")
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["Quick Comparison", "Detailed Analysis", "Risk Assessment", "Custom Query"]
        )
        
        if analysis_type == "Custom Query":
            custom_query = st.text_area(
                "Enter your custom analysis query:",
                placeholder="e.g., What are the growth prospects for these companies?"
            )

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock data visualization
        if selected_stocks:
            st.header(f"📈 Live Data for {', '.join(selected_stocks)}")
            
            with st.spinner("Fetching stock data..."):
                stock_data = get_stock_data(selected_stocks)
            
            if stock_data:
                # Current prices display
                cols = st.columns(len(selected_stocks))
                for i, (symbol, data) in enumerate(stock_data.items()):
                    with cols[i]:
                        if data['current_price']:
                            # Calculate daily change if possible
                            try:
                                prev_close = data['history']['Close'].iloc[-2]
                                current = data['current_price']
                                change = current - prev_close
                                change_pct = (change / prev_close) * 100
                                
                                st.metric(
                                    label=symbol,
                                    value=f"${current:.2f}",
                                    delta=f"{change:+.2f} ({change_pct:+.1f}%)"
                                )
                            except:
                                st.metric(
                                    label=symbol,
                                    value=f"${data['current_price']:.2f}"
                                )
                
                # Price chart
                st.plotly_chart(create_price_chart(stock_data), use_container_width=True)
                
                # Volume chart
                st.plotly_chart(create_volume_chart(stock_data), use_container_width=True)
    
    with col2:
        # Quick stats
        st.header("Quick Stats")
        if selected_stocks and stock_data:
            for symbol, data in stock_data.items():
                if data['info']:
                    with st.expander(f"{symbol} Info"):
                        info = data['info']
                        st.write(f"**Company:** {info.get('longName', 'N/A')}")
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Market Cap:** ${info.get('marketCap', 0):,.0f}")
                        st.write(f"**P/E Ratio:** {info.get('forwardPE', 'N/A')}")

    # AI Analysis Section
    st.header("AI Financial Analysis")
    
    if not st.session_state.agent:
        st.warning("⚠️ Please initialize the AI Agent in the sidebar first.")
    else:
        # Analysis execution
        if st.button("🔍 Run Analysis", type="primary", disabled=not selected_stocks):
            if selected_stocks:
                # Prepare query based on analysis type
                if analysis_type == "Quick Comparison":
                    query = f"""
                    Provide a quick comparison of {', '.join(selected_stocks)} stocks including:
                    - Current prices and recent performance
                    - Key financial metrics
                    - Brief analyst recommendations
                    Use tables for clear data presentation.
                    """
                elif analysis_type == "Detailed Analysis":
                    query = f"""
                    Perform a comprehensive analysis of {', '.join(selected_stocks)} including:
                    - Current stock prices and performance metrics
                    - Company financial health and ratios
                    - Analyst recommendations and price targets
                    - Recent company news and developments
                    - Investment risks and opportunities
                    Present findings in well-structured tables and provide detailed reasoning.
                    """
                elif analysis_type == "Risk Assessment":
                    query = f"""
                    Conduct a risk assessment for {', '.join(selected_stocks)} focusing on:
                    - Volatility and price stability
                    - Company-specific risks
                    - Market and sector risks
                    - Financial stability indicators
                    Provide risk ratings and mitigation strategies.
                    """
                else:  # Custom Query
                    query = f"Analyze {', '.join(selected_stocks)} stocks: {custom_query}"
                
                # Execute analysis
                with st.spinner("🧠 AI is analyzing the stocks..."):
                    try:
                        start_time = time.time()
                        response = st.session_state.agent.run(query)
                        end_time = time.time()
                        
                        # Display results
                        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                        st.markdown("### 📋 Analysis Results")
                        st.markdown(response.content)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'stocks': selected_stocks.copy(),
                            'query': query,
                            'response': response.content,
                            'duration': f"{end_time - start_time:.2f}s"
                        })
                        
                        st.success(f"✅ Analysis completed in {end_time - start_time:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Try reinitializing the agent or check your query.")

    # Analysis History
    if st.session_state.analysis_history:
        st.header("📚 Analysis History")
        
        with st.expander("View Previous Analyses"):
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Last 5
                st.markdown(f"**{analysis['timestamp']}** - {', '.join(analysis['stocks'])}")
                st.markdown(f"*Duration: {analysis['duration']}*")
                with st.expander(f"View Analysis {len(st.session_state.analysis_history) - i}"):
                    st.markdown(analysis['response'])
                st.divider()

    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This analysis is for educational purposes only. "
        "Always consult with a qualified financial advisor before making investment decisions."
    )

if __name__ == "__main__":
    main()