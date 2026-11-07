import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from product_writer.cli import main
from product_writer.core import (
    Product, ValidationError, block_catalog, candidates, load_products, markdown, render_card,
)
from product_writer.llm import APIError, LLMClient, generate

ROOT = Path(__file__).resolve().parents[1]


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.products = load_products((ROOT / 'examples/products.json').read_text(encoding='utf-8'))
        results = json.loads((ROOT / 'examples/result.groq.json').read_text(encoding='utf-8'))
        self.plans = []
        for product, item in zip(self.products, results):
            card = item['card']
            titles, queries = candidates(product)
            self.plans.append({
                'title_index': titles.index(card['seo_title']),
                **card['evidence'],
                'query_indices': [queries.index(q) for q in card['search_queries']],
            })

    def run_cli(self, args, plans=None):
        plans = self.plans if plans is None else plans
        def fake_generate(product, client):
            index = next(i for i, p in enumerate(self.products) if p.name == product.name)
            return render_card(product, plans[index])
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}), patch(
            'product_writer.cli.generate', side_effect=fake_generate
        ):
            return main(args)

    def test_two_cards_meet_contract(self):
        for product, plan in zip(self.products, self.plans):
            with self.subTest(product=product.name):
                card = render_card(product, plan)
                self.assertLessEqual(len(card['seo_title']), 70)
                self.assertTrue(800 <= len(card['seo_description']) <= 1200)
                self.assertTrue(3 <= len(card['search_queries']) <= 5)
                self.assertEqual(card['characteristics']['Степень защиты корпуса'], 'данных нет')
                for key, value in product.specs.items():
                    self.assertEqual(card['characteristics'][key], value)

    def test_bad_input_json(self):
        with self.assertRaisesRegex(ValidationError, 'Некорректный JSON'):
            load_products('{bad')

    def test_punctuation_only_description_does_not_crash(self):
        p = Product.parse({'name': 'Прибор', 'description': ';'})
        self.assertFalse(any(k.startswith('source_') for k in block_catalog(p)))
        with self.assertRaises(ValidationError):
            generate(p, LLMClient('test-key'))

    def test_missing_completeness_and_audience_reported(self):
        p = Product.parse({'name': 'Прибор'})
        self.assertTrue(any('полнота не проверена' in s for s in p.missing_fields()))
        self.assertIn('audience', p.missing_fields())
        self.assertIn('tasks', p.missing_fields())

    def test_commercial_answers_audience_and_tasks(self):
        card = render_card(self.products[0], self.plans[0])
        self.assertEqual(card['commercial_details']['audience'], self.products[0].audience)
        self.assertEqual(card['commercial_details']['tasks'], list(self.products[0].tasks))
        bad = {**self.plans[0], 'commercial_blocks': ['source_0']}
        with self.assertRaises(ValidationError):
            render_card(self.products[0], bad)

    def test_descriptive_titles_and_queries(self):
        card = render_card(self.products[1], self.plans[1])
        self.assertIn('GSM-модем', card['seo_title'])
        self.assertTrue(any('GSM-модем' in q for q in card['search_queries']))

    def test_schema_failure_isolated_in_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads((ROOT / 'examples/products.json').read_text(encoding='utf-8'))
            raw[1]['name'] = ''
            path = Path(tmp) / 'input.json'
            output = Path(tmp) / 'output.json'
            path.write_text(json.dumps(raw), encoding='utf-8')
            code = self.run_cli([str(path), '--output', str(output)])
            self.assertEqual(code, 1)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertIn('card', result[0])
            self.assertIn('error', result[1])

    def test_missing_commercial_data_is_explicit(self):
        from dataclasses import replace
        p = replace(self.products[0], audience='', tasks=())
        plan = {**self.plans[0], 'commercial_blocks': []}
        card = render_card(p, plan)
        self.assertIn('данных нет', card['commercial_details']['audience'])
        self.assertEqual(card['commercial_details']['tasks'], [])
        self.assertIn('tasks', card['missing_fields'])

    def test_short_source_returns_actionable_clarification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            output = Path(tmp) / 'output.json'
            path.write_text(json.dumps({'name': 'Прибор', 'description': ';'}), encoding='utf-8')
            with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}), patch(
                'product_writer.llm.urlopen'
            ) as network:
                self.assertEqual(main([str(path), '--output', str(output)]), 1)
                network.assert_not_called()
            error = json.loads(output.read_text(encoding='utf-8'))[0]
            self.assertEqual(error['status'], 'needs_clarification')
            self.assertTrue(any('description/specs' in f for f in error['missing_fields']))

    def test_spec_repeated_in_source_is_not_offered_to_model(self):
        blocks = block_catalog(self.products[0])
        self.assertNotIn('spec_0', blocks)
        self.assertIn('source_1', blocks)
        self.assertIn('spec_2', blocks)
        card = render_card(self.products[0], self.plans[0])
        self.assertEqual(card['characteristics']['Дискретные входы'],
                         '8 каналов, без гальванической развязки')

    def test_similar_but_different_value_is_preserved(self):
        product = Product.parse({'name': 'Прибор', 'specs': {'Питание': '24 В'},
                                 'description': 'Питание: 12 В.'})
        self.assertIn('spec_0', block_catalog(product))

    def test_bad_input_shapes(self):
        for raw in [None, [], 12, {'name': ''}, {'name': 'x', 'specs': []},
                    {'name': 'x', 'description': []}, {'name': 'x', 'required_specs': 'x'},
                    {'name': 'x', 'specs': {'power': 24}}]:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                load_products(json.dumps(raw))

    def test_missing_fields(self):
        p = Product.parse({'name': 'Прибор', 'specs': {'Питание': None},
                           'required_specs': ['Интерфейс']})
        self.assertEqual(p.characteristics(), {'Интерфейс': 'данных нет', 'Питание': 'данных нет'})
        self.assertIn('manufacturer', p.missing_fields())
        self.assertIn('description', p.missing_fields())

    def test_unknown_fact_and_free_text_rejected(self):
        for change in [{'description_blocks': ['Питание: 999 В']},
                       {'new_specs': {'Питание': '999 В'}},
                       {'title_index': -1}, {'query_indices': [0, 0, 1]},
                       {'description_blocks': ['name']},
                       {'description_blocks': ['name', 'name']},
                       {'query_indices': [True, 1, 2]},
                       {'commercial_blocks': ['editorial_0']}]:
            plan = {**self.plans[0], **change}
            with self.subTest(change=change), self.assertRaises(ValidationError):
                render_card(self.products[0], plan)

    def test_model_retry_after_invalid_plan(self):
        class Client:
            calls = 0

            def complete(inner, messages):
                inner.calls += 1
                return {} if inner.calls == 1 else self.plans[0]

        client = Client()
        generate(self.products[0], client)
        self.assertEqual(client.calls, 2)

    def test_model_repair_exhausted(self):
        with patch.object(LLMClient, 'complete', return_value={}) as complete:
            with self.assertRaisesRegex(ValidationError, 'за 3 попытки'):
                generate(self.products[0], LLMClient('test-key'))
            self.assertEqual(complete.call_count, 3)

    def test_sparse_input_no_api_call(self):
        with patch.object(LLMClient, 'complete') as complete:
            with self.assertRaisesRegex(ValidationError, 'Недостаточно данных'):
                generate(Product.parse({'name': 'Прибор'}), LLMClient('test-key'))
            complete.assert_not_called()

    def test_missing_key(self):
        with self.assertRaisesRegex(APIError, 'GROQ_API_KEY'):
            LLMClient('')

    @staticmethod
    def response(content, finish_reason='stop'):
        envelope = {'choices': [{'finish_reason': finish_reason,
                                 'message': {'content': content}}]}
        return io.BytesIO(json.dumps(envelope).encode())

    def test_http_request_contract(self):
        with patch('product_writer.llm.urlopen', return_value=self.response('{"ok":true}')) as call:
            result = LLMClient('secret').complete([{'role': 'user', 'content': 'JSON'}])
            self.assertTrue(result['ok'])
            request = call.call_args.args[0]
            self.assertEqual(request.get_header('Authorization'), 'Bearer secret')
            self.assertEqual(json.loads(request.data)['response_format'], {'type': 'json_object'})
            self.assertEqual(call.call_args.kwargs['timeout'], 40)

    def test_retry_transient_http(self):
        for code in (429, 503):
            error = HTTPError('https://api.groq.com', code, 'err', {}, None)
            with self.subTest(code=code), patch('product_writer.llm.time.sleep'), patch(
                'product_writer.llm.urlopen', side_effect=[error, self.response('{}')]
            ) as call:
                self.assertEqual(LLMClient('secret').complete([]), {})
                self.assertEqual(call.call_count, 2)

    def test_auth_error_no_retry_or_secret(self):
        with patch('product_writer.llm.urlopen', side_effect=HTTPError(
            'https://api.groq.com', 401, 'secret', {}, None
        )) as call:
            with self.assertRaises(APIError) as error:
                LLMClient('secret').complete([])
            self.assertNotIn('secret', str(error.exception))
            self.assertEqual(call.call_count, 1)

    def test_network_retries_bounded(self):
        with patch('product_writer.llm.time.sleep'), patch(
            'product_writer.llm.urlopen', side_effect=URLError('secret')
        ) as call:
            with self.assertRaisesRegex(APIError, 'тайм-аут'):
                LLMClient('secret').complete([])
            self.assertEqual(call.call_count, 3)

    def test_invalid_api_content(self):
        for response in [io.BytesIO(b'{}'), self.response('broken'),
                         self.response('{}', 'length'), io.BytesIO(b'not-json'),
                         io.BytesIO(b'{"choices":[null]}')]:
            with self.subTest(response=response), patch(
                'product_writer.llm.urlopen', return_value=response
            ), self.assertRaises(ValidationError):
                LLMClient('secret').complete([])

    def test_cli_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ('json', 'markdown'):
                output = Path(tmp) / f'card.{fmt}'
                code = self.run_cli([str(ROOT / 'examples/products.json'), '--format', fmt,
                                     '--output', str(output)])
                self.assertEqual(code, 0)
                content = output.read_text(encoding='utf-8')
                if fmt == 'json':
                    self.assertEqual(json.loads(content)[0]['mode'], 'llm_api')
                else:
                    self.assertIn('Задачи:', content)

    def test_whitespace_in_optional_fields(self):
        product = Product.parse({'name': 'x', 'manufacturer': ' ',
                                 'specs': {' Power ': ' '}, 'required_specs': [' Power ', 'Power']})
        self.assertEqual(product.manufacturer, 'данных нет')
        self.assertEqual(product.required_specs, ['Power'])
        self.assertEqual(product.characteristics(), {'Power': 'данных нет'})

    def test_ambiguous_json_is_rejected(self):
        for payload in ['{"name":"a","name":"b"}', '{"name":"x","value":NaN}']:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                load_products(payload)

    def test_signs_in_values_are_preserved_during_deduplication(self):
        p = Product.parse({'name': 'x', 'specs': {'Температура': '-10 C'},
                           'description': 'Температура +10 C.'})
        self.assertIn('spec_0', block_catalog(p))

    def test_missing_vendor_does_not_pollute_titles(self):
        p = Product.parse({'name': 'x', 'product_type': 'Манометр'})
        titles, queries = candidates(p)
        self.assertFalse(any('данных нет' in t for t in titles + queries))

    def test_custom_api_does_not_reuse_groq_key(self):
        with patch.dict('os.environ', {'GROQ_API_KEY': 'groq-secret', 'LLM_API_KEY': '',
                                      'LLM_MODEL': 'model'}):
            with self.assertRaises(APIError):
                LLMClient(base_url='https://example.org/v1')
        client = LLMClient('other-secret', 'other-model', 'https://example.org/v1/')
        self.assertEqual(client.endpoint, 'https://example.org/v1/chat/completions')

    def test_invalid_api_settings(self):
        for url in ['http://example.org/v1', 'https://user:password@example.org',
                    'https://[', 'https://example.org:invalid/v1']:
            with self.subTest(url=url), self.assertRaises(APIError):
                LLMClient('secret', 'model', url)
        with self.assertRaises(APIError):
            LLMClient('secret\ninvalid')

    def test_input_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.stderr', new_callable=io.StringIO):
            path = Path(tmp) / 'input.json'
            path.write_text('{"name":"x"}', encoding='utf-8')
            self.assertEqual(main([str(path), '--output', str(path)]), 2)
            self.assertEqual(path.read_text(encoding='utf-8'), '{"name":"x"}')

    def test_partial_batch_failure_preserves_success(self):
        plans = copy.deepcopy(self.plans)
        plans[0]['title_index'] = -1
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out.json'
            code = self.run_cli([str(ROOT / 'examples/products.json'),
                                 '--output', str(output)], plans=plans)
            self.assertEqual(code, 1)
            results = json.loads(output.read_text(encoding='utf-8'))
            self.assertIn('error', results[0])
            self.assertIn('card', results[1])
            self.assertIn('Ошибка', markdown(results))

    def test_cli_bad_file_and_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.stderr', new_callable=io.StringIO):
            path = Path(tmp) / 'invalid.json'
            self.assertEqual(main([str(path)]), 2)
            path.write_text('{', encoding='utf-8')
            self.assertEqual(main([str(path)]), 2)


if __name__ == '__main__':
    unittest.main()
