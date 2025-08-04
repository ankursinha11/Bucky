-- Sample SQL for CTM Mapping Test
-- This SQL demonstrates various transformations that should be captured in CTM mapping

SELECT 
    c.customer_id,
    c.customer_name,
    c.email,
    c.created_date,
    o.order_id,
    o.order_date,
    o.total_amount,
    CASE 
        WHEN o.total_amount > 1000 THEN 'High Value'
        WHEN o.total_amount > 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_segment,
    DATE_DIFF(o.order_date, c.created_date) as days_since_registration,
    SUM(o.total_amount) OVER (PARTITION BY c.customer_id) as total_customer_spend,
    ROW_NUMBER() OVER (PARTITION BY c.customer_id ORDER BY o.order_date) as order_sequence
FROM 
    customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN customer_preferences cp ON c.customer_id = cp.customer_id
WHERE 
    o.order_date >= '2023-01-01'
    AND o.status = 'completed'
    AND c.is_active = 1
GROUP BY 
    c.customer_id, c.customer_name, c.email, c.created_date,
    o.order_id, o.order_date, o.total_amount
HAVING 
    SUM(o.total_amount) > 100
ORDER BY 
    c.customer_id, o.order_date; 