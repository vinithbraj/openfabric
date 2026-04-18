import unittest

from agent_library.agents import sql_runner


SAMPLE_SCHEMA = {
    "dialect": "postgres",
    "tables": [
        {
            "schema": "flathr",
            "name": "study",
            "columns": [
                {"name": "patient_id", "type": "text", "nullable": False},
                {"name": "patient_name", "type": "text", "nullable": True},
                {"name": "study_uid", "type": "text", "nullable": False},
            ],
            "foreign_keys": [],
        },
        {
            "schema": "flathr",
            "name": "series",
            "columns": [
                {"name": "study_uid", "type": "text", "nullable": False},
            ],
            "foreign_keys": [],
        },
    ],
}


class SqlRunnerValidationTests(unittest.TestCase):
    def test_extract_table_references_handles_qualified_and_quoted_names(self):
        sql = 'SELECT * FROM "flathr"."study" s JOIN flathr.series se ON se.study_uid = s.study_uid'
        refs = sql_runner._extract_table_references(sql)
        self.assertEqual(refs, ["flathr.study", "flathr.series"])

    def test_validate_sql_against_schema_rejects_unknown_table(self):
        sql = 'SELECT patient_name, COUNT(*) FROM flathr.patient GROUP BY patient_name'
        with self.assertRaises(RuntimeError) as ctx:
            sql_runner._validate_sql_against_schema(sql, SAMPLE_SCHEMA)
        self.assertIn("flathr.patient", str(ctx.exception))
        self.assertIn("flathr.study", str(ctx.exception))

    def test_canonicalize_sql_identifiers_preserves_schema_case(self):
        schema = {
            "dialect": "postgres",
            "tables": [
                {
                    "schema": "FlatHR",
                    "name": "Patient",
                    "columns": [
                        {"name": "PatientID", "type": "text", "nullable": False},
                    ],
                    "foreign_keys": [],
                },
                {
                    "schema": "FlatHR",
                    "name": "Study",
                    "columns": [
                        {"name": "StudyInstanceUID", "type": "text", "nullable": False},
                        {"name": "PatientID", "type": "text", "nullable": False},
                    ],
                    "foreign_keys": [],
                },
            ],
        }
        sql = (
            "SELECT PatientID, COUNT(s.StudyInstanceUID) as StudyCount "
            "FROM flathr.patient p "
            "JOIN flathr.study s ON p.PatientID = s.PatientID "
            "GROUP BY PatientID "
            "ORDER BY StudyCount DESC "
            "LIMIT 10"
        )
        rewritten = sql_runner._canonicalize_sql_identifiers(sql, schema, "postgres")
        self.assertIn('FROM "FlatHR"."Patient" p', rewritten)
        self.assertIn('JOIN "FlatHR"."Study" s', rewritten)
        self.assertIn('COUNT(s."StudyInstanceUID")', rewritten)
        self.assertIn('p."PatientID" = s."PatientID"', rewritten)
        self.assertIn('SELECT "PatientID"', rewritten)
        self.assertIn('GROUP BY "PatientID"', rewritten)

    def test_schema_repair_hints_points_back_to_real_columns(self):
        hints = sql_runner._schema_repair_hints(
            SAMPLE_SCHEMA,
            "query the top 10 patients with most studies",
            'SELECT * FROM flathr.patient',
            'UndefinedTable: relation "flathr.patient" does not exist',
        )
        self.assertIn("flathr.study", hints)
        self.assertIn("patient_name", hints)

    def test_validate_sql_columns_against_schema_rejects_bad_join_column(self):
        schema = {
            "dialect": "postgres",
            "tables": [
                {
                    "schema": "flathr",
                    "name": "Patient",
                    "columns": [
                        {"name": "PatientID", "type": "text", "nullable": False},
                    ],
                    "foreign_keys": [],
                },
                {
                    "schema": "flathr",
                    "name": "RTPlan",
                    "columns": [
                        {"name": "SOPInstanceUID", "type": "text", "nullable": False},
                        {"name": "ReferencedStudyUID", "type": "text", "nullable": False},
                    ],
                    "foreign_keys": [],
                },
            ],
        }
        sql = (
            'SELECT p."PatientID" '
            'FROM "flathr"."Patient" p '
            'JOIN "flathr"."RTPlan" r ON p."PatientID" = r."PatientID"'
        )
        with self.assertRaises(RuntimeError) as ctx:
            sql_runner._validate_sql_columns_against_schema(sql, schema)
        self.assertIn("r.PatientID", str(ctx.exception))
        self.assertIn("SOPInstanceUID", str(ctx.exception))

    def test_resolve_and_compile_plan_uses_foreign_key_join(self):
        schema = {
            "dialect": "postgres",
            "tables": [
                {
                    "schema": "flathr",
                    "name": "Patient",
                    "columns": [
                        {"name": "PatientID", "type": "text", "nullable": False},
                    ],
                    "foreign_keys": [],
                },
                {
                    "schema": "flathr",
                    "name": "Study",
                    "columns": [
                        {"name": "StudyInstanceUID", "type": "text", "nullable": False},
                        {"name": "PatientID", "type": "text", "nullable": False},
                        {"name": "StudyDate", "type": "date", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "PatientID",
                            "references_schema": "flathr",
                            "references_table": "Patient",
                            "references_column": "PatientID",
                        }
                    ],
                },
            ],
        }
        plan = {
            "summary": "patient study counts",
            "tables": ["flathr.patient", "flathr.study"],
            "dimensions": [{"table": "flathr.patient", "column": "patientid", "alias": "patient_id"}],
            "measures": [
                {"table": "flathr.study", "column": "studyinstanceuid", "aggregate": "count", "alias": "study_count"},
                {"table": "flathr.study", "column": "studydate", "aggregate": "max", "alias": "most_recent_study"},
            ],
            "joins": [
                {
                    "left_table": "flathr.study",
                    "left_column": "patientid",
                    "right_table": "flathr.patient",
                    "right_column": "patientid",
                    "join_type": "inner",
                }
            ],
            "filters": [],
            "group_by": [{"table": "flathr.patient", "column": "patientid"}],
            "having": [{"measure_alias": "study_count", "operator": ">", "value": 2}],
            "order_by": [{"type": "measure", "measure_alias": "study_count", "direction": "desc"}],
            "limit": 10,
        }
        resolved = sql_runner._resolve_sql_plan(plan, schema)
        sql = sql_runner._compile_sql_from_plan(resolved, schema, "postgres")
        self.assertIn('FROM "flathr"."Patient" t1', sql)
        self.assertIn('JOIN "flathr"."Study" t2', sql)
        self.assertIn('t1."PatientID"', sql)
        self.assertIn('COUNT(t2."StudyInstanceUID") AS "study_count"', sql)
        self.assertIn('MAX(t2."StudyDate") AS "most_recent_study"', sql)
        self.assertIn('HAVING COUNT(t2."StudyInstanceUID") > 2', sql)

    def test_compile_sql_from_plan_rejects_unjoined_referenced_tables(self):
        schema = {
            "dialect": "postgres",
            "tables": [
                {
                    "schema": "flathr",
                    "name": "Patient",
                    "columns": [{"name": "PatientID", "type": "text", "nullable": False}],
                    "foreign_keys": [],
                },
                {
                    "schema": "flathr",
                    "name": "MatrixRegistrationSequence",
                    "columns": [{"name": "MatrixRegistrationSequenceID", "type": "text", "nullable": False}],
                    "foreign_keys": [],
                },
            ],
        }
        plan = {
            "tables": ["flathr.Patient", "flathr.MatrixRegistrationSequence"],
            "dimensions": [
                {"table": "flathr.Patient", "column": "PatientID", "alias": "patient_id"},
                {
                    "table": "flathr.MatrixRegistrationSequence",
                    "column": "MatrixRegistrationSequenceID",
                    "alias": "matrix_registration_sequence_id",
                },
            ],
            "measures": [],
            "joins": [],
            "filters": [],
            "group_by": [],
            "having": [],
            "order_by": [],
            "limit": 10,
        }
        with self.assertRaises(RuntimeError) as ctx:
            sql_runner._compile_sql_from_plan(plan, schema, "postgres")
        self.assertIn("not connected by valid joins", str(ctx.exception))

    def test_preferred_query_task_uses_original_request_when_instruction_is_generic(self):
        instruction = {"operation": "query_from_request", "question": "query the database for the required information"}
        execution_task = "Current workflow step: query the database for the required information"
        original_task = "from mydb\nFor each patient who has undergone radiation therapy, retrieve their patient ID."
        preferred = sql_runner._preferred_query_task(instruction, execution_task, original_task)
        self.assertEqual(preferred, original_task)

    def test_infer_join_tree_connects_rt_plan_to_patient_via_series_and_study(self):
        schema = {
            "dialect": "postgres",
            "tables": [
                {
                    "schema": "flathr",
                    "name": "Patient",
                    "columns": [{"name": "PatientID", "type": "text", "nullable": False}],
                    "foreign_keys": [],
                },
                {
                    "schema": "flathr",
                    "name": "Study",
                    "columns": [
                        {"name": "StudyInstanceUID", "type": "text", "nullable": False},
                        {"name": "PatientID", "type": "text", "nullable": True},
                        {"name": "StudyDate", "type": "date", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "PatientID",
                            "references_schema": "flathr",
                            "references_table": "Patient",
                            "references_column": "PatientID",
                        }
                    ],
                },
                {
                    "schema": "flathr",
                    "name": "Series",
                    "columns": [
                        {"name": "SeriesInstanceUID", "type": "text", "nullable": False},
                        {"name": "StudyInstanceUID", "type": "text", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "StudyInstanceUID",
                            "references_schema": "flathr",
                            "references_table": "Study",
                            "references_column": "StudyInstanceUID",
                        }
                    ],
                },
                {
                    "schema": "flathr",
                    "name": "RTPlan",
                    "columns": [
                        {"name": "SOPInstanceUID", "type": "text", "nullable": False},
                        {"name": "SeriesInstanceUID", "type": "text", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "SeriesInstanceUID",
                            "references_schema": "flathr",
                            "references_table": "Series",
                            "references_column": "SeriesInstanceUID",
                        }
                    ],
                },
                {
                    "schema": "flathr",
                    "name": "DoseReferenceSequence",
                    "columns": [
                        {"name": "DoseReferenceSequenceID", "type": "bigint", "nullable": False},
                        {"name": "SOPInstanceUID", "type": "text", "nullable": True},
                        {"name": "TargetPrescriptionDose", "type": "double precision", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "SOPInstanceUID",
                            "references_schema": "flathr",
                            "references_table": "RTPlan",
                            "references_column": "SOPInstanceUID",
                        }
                    ],
                },
            ],
        }
        tables, joins = sql_runner._infer_join_tree(
            ["flathr.Patient", "flathr.RTPlan", "flathr.DoseReferenceSequence"],
            schema,
        )
        self.assertIn("flathr.Series", tables)
        self.assertIn("flathr.Study", tables)
        join_pairs = {(join["left_table"], join["right_table"]) for join in joins}
        self.assertIn(("flathr.Study", "flathr.Patient"), join_pairs)
        self.assertIn(("flathr.Series", "flathr.Study"), join_pairs)
        self.assertIn(("flathr.RTPlan", "flathr.Series"), join_pairs)
        self.assertIn(("flathr.DoseReferenceSequence", "flathr.RTPlan"), join_pairs)

    def test_resolve_sql_plan_recovers_numeric_dose_measure_and_schema_join_path(self):
        schema = {
            "dialect": "postgres",
            "tables": [
                {
                    "schema": "flathr",
                    "name": "Patient",
                    "columns": [{"name": "PatientID", "type": "text", "nullable": False}],
                    "foreign_keys": [],
                },
                {
                    "schema": "flathr",
                    "name": "Study",
                    "columns": [
                        {"name": "StudyInstanceUID", "type": "text", "nullable": False},
                        {"name": "PatientID", "type": "text", "nullable": True},
                        {"name": "StudyDate", "type": "date", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "PatientID",
                            "references_schema": "flathr",
                            "references_table": "Patient",
                            "references_column": "PatientID",
                        }
                    ],
                },
                {
                    "schema": "flathr",
                    "name": "Series",
                    "columns": [
                        {"name": "SeriesInstanceUID", "type": "text", "nullable": False},
                        {"name": "StudyInstanceUID", "type": "text", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "StudyInstanceUID",
                            "references_schema": "flathr",
                            "references_table": "Study",
                            "references_column": "StudyInstanceUID",
                        }
                    ],
                },
                {
                    "schema": "flathr",
                    "name": "RTPlan",
                    "columns": [
                        {"name": "SOPInstanceUID", "type": "text", "nullable": False},
                        {"name": "SeriesInstanceUID", "type": "text", "nullable": True},
                        {"name": "PlanIntent", "type": "text", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "SeriesInstanceUID",
                            "references_schema": "flathr",
                            "references_table": "Series",
                            "references_column": "SeriesInstanceUID",
                        }
                    ],
                },
                {
                    "schema": "flathr",
                    "name": "DoseReferenceSequence",
                    "columns": [
                        {"name": "DoseReferenceSequenceID", "type": "bigint", "nullable": False},
                        {"name": "SOPInstanceUID", "type": "text", "nullable": True},
                        {"name": "TargetPrescriptionDose", "type": "double precision", "nullable": True},
                    ],
                    "foreign_keys": [
                        {
                            "column": "SOPInstanceUID",
                            "references_schema": "flathr",
                            "references_table": "RTPlan",
                            "references_column": "SOPInstanceUID",
                        }
                    ],
                },
            ],
        }
        plan = {
            "summary": "rt analytics",
            "tables": ["flathr.RTPlan", "flathr.RTDOSE", "flathr.Study", "flathr.Patient"],
            "dimensions": [{"table": "flathr.Patient", "column": "PatientID", "alias": "PatientID"}],
            "measures": [
                {"table": "flathr.RTPlan", "column": "count", "aggregate": "count", "alias": "NumberOfRTPlans"},
                {"table": "flathr.RTDOSE", "column": "DoseSummationType", "aggregate": "avg", "alias": "AverageDose"},
                {"table": "flathr.Study", "column": "StudyDate", "aggregate": "max", "alias": "MostRecentStudyDate"},
            ],
            "joins": [],
            "filters": [{"table": "flathr.RTPlan", "column": "PlanIntent", "operator": "in", "value": ["RT", "RP"]}],
            "group_by": [{"table": "flathr.Patient", "column": "PatientID"}],
            "having": [{"measure_alias": "NumberOfRTPlans", "operator": ">", "value": 2}],
            "order_by": [{"type": "measure", "measure_alias": "AverageDose", "direction": "desc"}],
            "limit": 10,
        }
        resolved = sql_runner._resolve_sql_plan(
            plan,
            schema,
            task="For each patient, average radiation dose delivered across plans and most recent study date.",
        )
        measure_map = {item["alias"]: item for item in resolved["measures"]}
        self.assertEqual(measure_map["AverageDose"]["table"], "flathr.DoseReferenceSequence")
        self.assertEqual(measure_map["AverageDose"]["column"], "TargetPrescriptionDose")
        self.assertEqual(measure_map["NumberOfRTPlans"]["aggregate"], "count_distinct")
        sql = sql_runner._compile_sql_from_plan(resolved, schema, "postgres")
        self.assertIn('"TargetPrescriptionDose") AS "AverageDose"', sql)
        self.assertIn('COUNT(DISTINCT', sql)
        self.assertIn('"SOPInstanceUID") AS "NumberOfRTPlans"', sql)
        self.assertIn('HAVING COUNT(DISTINCT', sql)


if __name__ == "__main__":
    unittest.main()
