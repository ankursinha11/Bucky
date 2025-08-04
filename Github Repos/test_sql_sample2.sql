-- Verified Income Analysis Query
-- This query extracts comprehensive verified income data with related loan, customer, and verification information

SELECT 
    'avant' as partner,
    v.id as verified_income_id,
    l.id as loan_id,
    v.object_uuid as loan_uuid,
    v.object_type,
    c.id as customer_id,
    v.customer_uuid,
    calc.vtl_id as vtl_id,
    l.customer_application_id as customer_application_id,
    v.customer_application_uuid,
    au.email as admin_user_email,
    au.admin_user_id,
    v.admin_user_uuid,
    v.total_amount_cents as total_amount_cents,
    tzcorrect(v.created_at) as verified_income_calculation_created_at,
    v.updated_at,
    v.outcome as outcome,
    v.claimed_income as claimed_income,
    v.auto_calculated,
    CASE 
        WHEN (v.simulated_terms != '{}') OR (v.simulated_terms IS NOT NULL) 
        THEN TRUE 
        ELSE FALSE 
    END as simulated_terms_flag,
    v.similated_terms,
    v.simulated_terms_table,
    er.result as error_reason,
    CASE WHEN v.outcome = 'rejected' THEN TRUE ELSE FALSE END as rejected_flag,
    CASE WHEN er.result IS NOT NULL THEN TRUE ELSE FALSE END as error_reason_flag,
    CASE WHEN n.note_id IS NOT NULL THEN TRUE ELSE FALSE END as documentation_error_flag,
    calc.vt_name as verification_task_name,
    CASE 
        WHEN au.id = 7 THEN 'system_calculation'
        WHEN calc.vt_name = 'team_lead_enter_net_inc' THEN 'tl_review'
        WHEN calc.vt_name = 'enter_net_inc' THEN 'specialist_calculation'
        ELSE NULL 
    END as calculation_type,
    calc.reviewed_by as reviewed_by_calc_id,
    calc.reviewed as reviewed_calc_id,
    ird.ir_document,
    n_error.error_text as team_lead_income_discrepancy_error_note,
    CASE 
        WHEN n.note_id IS NOT NULL 
        THEN nde.error_text_doc 
        ELSE NULL 
    END as documentation_error_note,
    v.original_apr_percentage,
    v.original_loan_amount_cents,
    v.original_term,
    v.data

FROM verified_incomes v
LEFT JOIN loans l ON l.uuid = CAST(v.object_uuid AS VARCHAR)
LEFT JOIN customers c ON CAST(v.customer_uuid AS VARCHAR) = c.uuid
LEFT JOIN admin_users au ON CAST(v.admin_user_uuid AS VARCHAR) = au.uuid

-- Subquery to get verification task calculations with review relationships
LEFT JOIN (
    SELECT 
        vtl.id as vtl_id,
        vi.id as vi_id,
        vt.name as vt_name,
        CASE 
            WHEN ((vt.name = 'enter_net_inc') AND 
                  (LAG(vt.name) OVER(PARTITION BY l.id ORDER BY vtl.created_at DESC) = 'team_lead_enter_net_inc'))
            THEN LAG(vi.id) OVER(PARTITION BY l.id ORDER BY vtl.created_at DESC)
            ELSE NULL 
        END as reviewed_by,
        CASE 
            WHEN ((vt.name = 'team_lead_enter_net_inc') AND 
                  (LEAD(vt.name) OVER(PARTITION BY l.id ORDER BY vtl.created_at DESC) = 'enter_net_inc'))
            THEN LEAD(vi.id) OVER(PARTITION BY l.id ORDER BY vtl.created_at DESC)
            ELSE NULL 
        END as reviewed
    FROM verified_incomes vi
    JOIN loans l ON l.uuid = CAST(vi.object_uuid AS VARCHAR)
    INNER JOIN admin_users au ON au.uuid = CAST(vi.admin_user_uuid AS VARCHAR)
    LEFT JOIN verification_task_logs vtl ON vtl.id = (
        SELECT vtl.id
        FROM verification_task_logs vtl
        JOIN verification_tasks vt ON vt.id = vtl.verification_task_id 
            AND vt.product_type = 'Loan'
        WHERE vt.product_uuid = l.uuid
            AND CONCAT('admin_', au.id, ' ') = vtl.whodunnit
            AND vtl.created_at >= vi.created_at
            AND vtl.created_at <= (vi.created_at + INTERVAL '10 minutes')
            AND SPLIT_PART(vtl.whodunnit, '_', 1) = 'admin'
        ORDER BY vtl.created_at ASC
        LIMIT 1
    )
    INNER JOIN verification_tasks vt ON vt.id = vtl.verification_task_id
        AND (vt.name = 'enter_net_inc' OR vt.name = 'team_lead_enter_net_inc')
    WHERE vi.created_at > CAST('2018-07-01' AS DATE)
) calc ON calc.vi_id = v.id

-- Subquery to get error reasons from team lead verification tasks
LEFT JOIN (
    SELECT
        vtl.result as result,
        vi.id as vi_id
    FROM verified_incomes vi
    INNER JOIN admin_users au ON au.uuid = CAST(vi.admin_user_uuid AS VARCHAR)
    JOIN loans l ON l.uuid = CAST(vi.object_uuid AS VARCHAR)
    INNER JOIN verification_task_logs vtl ON vtl.id = (
        SELECT vtl.id
        FROM verification_task_logs vtl
        JOIN verification_tasks vt ON vt.id = vtl.verification_task_id 
            AND vt.product_type = 'Loan'
        WHERE vt.product_uuid = l.uuid
            AND CONCAT('admin_', au.id, ' ') = vtl.whodunnit
            AND vtl.created_at >= vi.created_at
            AND vtl.created_at <= (vi.created_at + INTERVAL '10 minutes')
            AND SPLIT_PART(vtl.whodunnit, '_', 1) = 'admin'
            AND vt.name = 'team_lead_identify_income_calculation_error' 
            AND vtl.result != ''
        ORDER BY vtl.created_at ASC
        LIMIT 1
    )
    LEFT JOIN verification_tasks vt ON vt.id = vtl.verification_task_id
    WHERE vi.created_at > CAST('2018-07-01' AS DATE)
) er ON calc.reviewed_by = er.vi_id

-- Subquery to identify documentation errors from notes
LEFT JOIN (
    SELECT 
        vi.id as verified_income_id,
        n.id as note_id,
        ROW_NUMBER() OVER (PARTITION BY vi.id ORDER BY n.created_at ASC) as row_num
    FROM (
        SELECT 
            id,
            created_at,
            LEAD(created_at) OVER (PARTITION BY object_uuid ORDER BY created_at) as next_vi_created_at,
            object_uuid
        FROM verified_incomes
        WHERE created_at > CAST('2018-07-01' AS DATE)
    ) vi
    JOIN loans l ON l.uuid = CAST(vi.object_uuid AS VARCHAR)
    JOIN notes n ON n.notable_id = l.id
        AND n.notable_type = 'Loan'
        AND n.note_type = 'Verifications'
        AND n.note_action = 'Team Lead Review'
        AND n.note_activity = 'Income Amount'
        AND n.text LIKE '%Existing documentation does not adequately verify income. Income capture tasks have been reset.%'
        AND n.created_at < COALESCE(vi.next_vi_created_at, NOW())
        AND n.created_at >= vi.created_at
) n ON n.verified_income_id = v.id AND n.row_num = 1

-- Subquery to get team lead income discrepancy error notes
LEFT JOIN (
    SELECT 
        vi.id as verified_income_id,
        n.id as note_id,
        n.text as error_text,
        ROW_NUMBER() OVER (PARTITION BY vi.id ORDER BY n.created_at ASC) as row_num
    FROM (
        SELECT 
            id,
            created_at,
            LEAD(created_at) OVER (PARTITION BY object_uuid ORDER BY created_at) as next_vi_created_at,
            object_uuid
        FROM verified_incomes
        WHERE created_at > CAST('2018-07-01' AS DATE)
    ) vi
    JOIN loans l ON l.uuid = CAST(vi.object_uuid AS VARCHAR)
    JOIN notes n ON n.notable_id = l.id
        AND n.notable_type = 'Loan'
        AND n.note_type = 'Verifications'
        AND n.note_activity = 'Team Lead Identify Income Calculation Error'
        AND n.created_at < COALESCE(vi.next_vi_created_at, NOW())
        AND n.created_at >= vi.created_at
) n_error ON n_error.verified_income_id = calc.reviewed_by AND n_error.row_num = 1

-- Subquery to get income document types
LEFT JOIN (
    SELECT 
        LAG(vt.name) OVER (PARTITION BY vt.product_uuid, vtl.whodunnit ORDER BY vtl.created_at) as ir_document,
        vtl.id as vtl_id
    FROM verification_task_logs vtl
    JOIN verification_tasks vt ON vt.id = vtl.verification_task_id 
        AND vtl.whodunnit LIKE '%admin%'
        AND vt.name IN (
            'enter_net_inc', 
            'team_lead_enter_net_inc', 
            'doc_bank_statement', 
            'yodlee_inc',
            'doc_paystub',
            'doc_tax_return',
            'doc_bank_statement_and_benefits_letter'
        )
) ird ON ird.vtl_id = calc.vtl_id

-- Subquery to get documentation error notes
LEFT JOIN (
    SELECT 
        vi.id as verified_income_id,
        n.id as note_id,
        n.text as error_text_doc,
        n.note_action as error_message,
        ROW_NUMBER() OVER (PARTITION BY vi.id ORDER BY n.created_at ASC) as row_num
    FROM (
        SELECT 
            id,
            created_at,
            LEAD(created_at) OVER (PARTITION BY object_uuid ORDER BY created_at) as next_vi_created_at,
            object_uuid
        FROM verified_incomes
        WHERE created_at > CAST('2018-07-01' AS DATE)
    ) vi
    JOIN loans l ON l.uuid = CAST(vi.object_uuid AS VARCHAR)
    JOIN notes n ON n.notable_id = l.id
        AND n.notable_type = 'Loan'
        AND n.note_type = 'Verifications'
        AND n.note_activity LIKE '%Doc%'
        AND n.created_at < COALESCE(vi.next_vi_created_at, NOW())
        AND n.created_at >= vi.created_at
) nde ON nde.verified_income_id = v.id AND nde.row_num = 1

WHERE LOWER(v.object_type) = 'loan';