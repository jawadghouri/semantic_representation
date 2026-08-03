import numpy as np
import matplotlib.pyplot as plt

# 1. Dummy Data: Replace these with your actual 384-dimensional vector embeddings
# (Assuming they are un-normalized raw embeddings based on your previous logs)
np.random.seed(42) 
embedding_streams = np.cumsum(np.random.normal(0.1, 0.05, 100))    # Simulated rising curve
embedding_downloads = np.cumsum(np.random.normal(-0.02, 0.08, 100)) # Simulated hump curve
embedding_cds = np.cumsum(np.random.normal(-0.1, 0.04, 100))       # Simulated declining curve

# Create an X-axis matching the number of features/dimensions in your vectors
dimensions = list(range(len(embedding_streams)))

# 2. Initialize the plot
plt.figure(figsize=(9, 5))

# 3. Plot each embedding with customized line styles and widths to match the image
# Streams: Dashed blue line
plt.plot(dimensions, embedding_streams, label='Streams', 
         color='#0066CC', linestyle='--', linewidth=3.5)

# Downloads: Solid red line
plt.plot(dimensions, embedding_downloads, label='Downloads', 
         color='#FF0000', linestyle='-', linewidth=3.5)

# CDs purchased: Dotted green line
plt.plot(dimensions, embedding_cds, label='CDs purchased', 
         color='#8BC34A', linestyle=':', linewidth=3.5)

# 4. Graph Titles and Axis styling
plt.title('Percentage of total music sales by method', fontsize=14, fontweight='bold', loc='left', pad=25)
plt.text(0, 1.02, 'Percentage', transform=plt.gca().transAxes, fontsize=12, color='#000000')

# Clear borders (spines) to match the clean canvas style
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# Add subtle horizontal gridlines
plt.grid(axis='y', linestyle='-', linewidth=0.5, color='#E0E0E0')

# Rotate X-axis labels to match the slanted text in your image
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

# 5. Position the legend on the right side outside the plot box
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, handlelength=2.5, fontsize=12)

# 6. Save and clean up
plt.tight_layout()
plt.savefig('music_sales_embeddings.png', dpi=300, bbox_inches='tight')
plt.show()