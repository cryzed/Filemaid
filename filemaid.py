#!/usr/bin/env python

import abc
import argparse
import functools
import itertools
import operator
import os
import re
import shutil
import sys
import textwrap
from datetime import datetime, timedelta

import magic
import yaml

SUCCESS_EXIT_STATUS = 0
FAILURE_EXIT_STATUS = 1
ENCODING = 'UTF-8'

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('rules')
argument_parser.add_argument('path')
argument_parser.add_argument('--dry-run', '-d', action='store_true')
argument_parser.add_argument('--recursive', '-r', action='store_true')
argument_parser.add_argument('--follow-symlinks', '-s', action='store_true')


class ConditionError(Exception):
    pass


class ActionError(Exception):
    pass


class Matchable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def match(self, path):
        pass


class Rule(Matchable):
    def __init__(self, name, condition, actions):
        self.name = name
        self.condition = condition
        self.actions = actions

        # Accumulate all action ignore paths to prevent Filemaid applying rules on action destination folders and such
        self.ignore_paths = set(itertools.chain(*(action.ignore_paths for action in actions)))

    def match(self, path):
        return self.condition.match(path)

    def apply(self, path):
        for action in self.actions:
            path = action.apply(path) or path

        return path

    def __repr__(self):
        condition_repr = repr(self.condition)
        actions_repr = repr(self.actions)
        return f'Rule(\n{textwrap.indent(condition_repr, "    ")},\n{textwrap.indent(actions_repr, "    ")}\n)'


class BaseTermCondition(Matchable):
    def __init__(self, *condition_data):
        super().__init__()
        self.conditions = [make_condition(datum) for datum in condition_data]

    def __repr__(self):
        condition_reprs = ',\n'.join(repr(condition) for condition in self.conditions)
        return f'{self.__class__.__name__}(\n{textwrap.indent(condition_reprs, "    ")}\n)'


class AllCondition(BaseTermCondition):
    def match(self, path):
        return all(condition.match(path) for condition in self.conditions)


class AnyCondition(BaseTermCondition):
    def match(self, path):
        return any(condition.match(path) for condition in self.conditions)


class NotCondition(Matchable):
    def __init__(self, condition_datum):
        super().__init__()
        self.condition = make_condition(condition_datum)

    def match(self, path):
        return not self.condition.match(path)

    def __repr__(self):
        return f'NotCondition({repr(self.condition)})'


class PathCondition(Matchable):
    def __init__(self, regex):
        super().__init__()
        self.regex = re.compile(regex)

    def match(self, path):
        return bool(self.regex.match(path))

    def __repr__(self):
        return f'PathCondition({repr(self.regex.pattern)})'


class MimeCondition(Matchable):
    def __init__(self, regex, ignore_case=True, magic_bytes=1024):
        super().__init__()
        self.regex = re.compile(regex, re.IGNORECASE if ignore_case else 0)
        self.magic_bytes = magic_bytes

    @functools.lru_cache(None)
    def match(self, path):
        if not os.path.isfile(path):
            return False

        with open(path, 'rb') as file:
            buffer = file.read(self.magic_bytes)
            mime = magic.from_buffer(buffer, mime=True)

        return bool(self.regex.match(mime))

    def __repr__(self):
        return f'MimeCondition({repr(self.regex.pattern)}, {repr(self.magic_bytes)})'


class AgeCondition(Matchable):
    UNITS = {'seconds', 'minutes', 'hours', 'days', 'weeks'}
    COMPARATORS = {
        '>': operator.gt,
        '>=': operator.ge,
        '=': operator.eq,
        '<=': operator.le,
        '<': operator.lt
    }

    def __init__(self, condition_string):
        self.condition_string = condition_string
        self.age_predicate = self.parse_age_condition(condition_string)

    def parse_age_condition(self, condition_string):
        comparator_string, size_string, unit_string = condition_string.split()
        compare = self.COMPARATORS[comparator_string]
        kwargs = {unit_string.lower(): int(size_string)}
        time_delta = timedelta(**kwargs)
        return lambda other: compare(datetime.now() - other, time_delta)

    def match(self, path):
        stat = os.stat(path)
        date_time = datetime.fromtimestamp(stat.st_mtime)
        return self.age_predicate(date_time)

    def __repr__(self):
        return f'AgeCondition({repr(self.condition_string)})'


class SizeCondition(Matchable):
    UNITS = {
        'b': 1,
        'kb': 1024,
        'mb': 1024 ** 2,
        'gb': 1024 ** 3,
        'tb': 1024 ** 4,
        'kib': 1000,
        'mib': 1000 ** 2,
        'gib': 1000 ** 3,
        'tib': 1000 ** 4
    }
    COMPARATORS = {
        '>': operator.gt,
        '>=': operator.ge,
        '=': operator.eq,
        '<=': operator.le,
        '<': operator.lt
    }

    def __init__(self, size):
        self.size_string = size
        self.size_predicate = self.parse_size_condition(size)

    def parse_size_condition(self, condition_string):
        comparator_string, size_string, unit_string = condition_string.split()
        size = self.UNITS[unit_string.lower()] * float(size_string)
        compare = self.COMPARATORS[comparator_string]
        return lambda other: compare(other, size)

    def match(self, path):
        stat = os.stat(path)
        return self.size_predicate(stat.st_size)

    def __repr__(self):
        return f'SizeCondition({repr(self.size_string)})'


class BaseAction(metaclass=abc.ABCMeta):
    def __init__(self):
        self.ignore_paths = set()

    @abc.abstractmethod
    def apply(self, path):
        # return the new path if modified
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class MoveAction(BaseAction):
    def __init__(self, destination):
        super().__init__()
        self.destination = os.path.expanduser(destination)
        self.ignore_paths.add(self.destination)

    def apply(self, path):
        os.makedirs(self.destination, exist_ok=True)
        return shutil.move(path, self.destination)

    def __repr__(self):
        return f'MoveAction({repr(self.destination)})'


class CopyAction(BaseAction):
    def __init__(self, destination):
        super().__init__()
        self.destination = os.path.expanduser(destination)
        self.ignore_paths.add(self.destination)

    def apply(self, path):
        os.makedirs(self.destination, exist_ok=True)
        shutil.copy2(path, self.destination)

    def __repr__(self):
        return f'CopyAction({repr(self.destination)})'


class DeleteAction(BaseAction):
    def apply(self, path):
        os.remove(path)


CONDITIONS = {
    'all': AllCondition,
    'any': AnyCondition,
    'not': NotCondition,
    'path': PathCondition,
    'mime': MimeCondition,
    'age': AgeCondition,
    'size': SizeCondition
}

ACTIONS = {
    'move': MoveAction,
    'copy': CopyAction,
    'delete': DeleteAction
}


def make_condition(data):
    if isinstance(data, dict):
        type_, data = list(data.items())[0]
    else:
        type_, data = data, []

    class_ = CONDITIONS.get(type_)
    if not class_:
        raise ConditionError(f'unknown type: {type_}')

    # arguments are keyword-arguments
    if isinstance(data, dict):
        return class_(**data)

    # arguments are positonal arguments
    if isinstance(data, list):
        return class_(*data)

    # a single argument
    return class_(data)


def make_actions(data):
    actions = []
    for datum in data:
        if isinstance(datum, dict):
            type_, datum = list(datum.items())[0]
        else:
            type_, datum = datum, []

        class_ = ACTIONS.get(type_)
        if not class_:
            raise ActionError(f'unknown type: {type_}')

        # arguments are keyword-arguments
        if isinstance(datum, dict):
            action = class_(**datum)
        # arguments are positonal arguments
        elif isinstance(datum, list):
            action = class_(*datum)
        # a single argument
        else:
            action = class_(datum)
        actions.append(action)

    return actions


def make_rule(data):
    name, data = list(data.items())[0]
    condition = make_condition(data['condition'])
    actions = make_actions(data['actions'])
    return Rule(name, condition, actions)


def load_rules(path):
    with open(path, encoding=ENCODING) as file:
        config = yaml.load(file)

    # Positional priority
    return [make_rule(data) for data in config]


def find_paths(path, predicate=lambda path: True, recursive=True, follow_symlinks=False):
    for root, directories, file_names in os.walk(path, followlinks=follow_symlinks):
        for name in itertools.chain(directories, file_names, [root]):
            path = os.path.join(root, name)
            if predicate(path):
                yield path

        if not recursive:
            break


def main(arguments):
    if not os.path.isfile(arguments.rules):
        print(f'No such file: {arguments.rules}', file=sys.stderr)
        return FAILURE_EXIT_STATUS

    if not os.path.isdir(arguments.path):
        print(f'No such folder: {arguments.path}', file=sys.stderr)
        return FAILURE_EXIT_STATUS

    rules = load_rules(arguments.rules)
    paths = find_paths(arguments.path, recursive=arguments.recursive, follow_symlinks=arguments.follow_symlinks)
    ignore_paths = {os.path.abspath(arguments.path)}
    ignore_paths.update(itertools.chain(*(rule.ignore_paths for rule in rules)))
    for path in (path for path in paths if path not in ignore_paths):
        for rule in rules:
            if rule.match(path):
                if arguments.dry_run:
                    print(f'{rule.name}: {path}')
                    continue

                # Potentially updated path, not used as of yet
                path = rule.apply(path)
                # Break after the first rule matched. Priority is given by order of appearance
                break

    return SUCCESS_EXIT_STATUS


if __name__ == '__main__':
    arguments = argument_parser.parse_args()
    argument_parser.exit(main(arguments))
