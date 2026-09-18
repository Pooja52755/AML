import fraud_data
import graph_vis

print('Testing fraud_data...')
txs = fraud_data.get_flagged_transactions()
print(f'  Flagged txs: {len(txs)}')

rows = fraud_data.get_fan_out_rows('TX-10231')
print(f'  Fan-out rows TX-10231: {len(rows)}')

senders = fraud_data.get_all_flagged_senders()
print(f'  Unique flagged senders: {len(senders)}')

rp = fraud_data.get_receiver_profile('ACCT-801')
name = rp['name']
print(f'  Receiver ACCT-801 profile: {name}')

print('Testing graph_vis...')
fig = graph_vis.render_plotly_graph('TX-10231', include_2hop=True)
print(f'  Graph traces count: {len(fig.data)}')

fig2 = graph_vis.render_plotly_graph('TX-10228', include_2hop=True)
print(f'  Graph traces TX-10228: {len(fig2.data)}')

print('All checks PASSED!')
