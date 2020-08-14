# -*- coding: utf-8 -*-
import argparse

from pyscaffold.api import Extension, helpers
from pyscaffold.extensions.no_skeleton import NoSkeleton
from pyscaffold.extensions.pre_commit import PreCommit

from pyscaffoldext.markdown.extension import MarkDown

from . import templates


class IncludeExtensions(argparse.Action):
    """Activate other extensions
    """

    def __call__(self, parser, namespace, values, option_string=None):
        print(parser)
        extensions = [
            NoSkeleton("no_skeleton"),
            PreCommit("pre_commit"),
            PytorchProject("pytorch"),
        ]
        namespace.extensions.extend(extensions)


class PytorchProject(Extension):
    """Template for data-science projects
    """

    def augment_cli(self, parser):
        """Augments the command-line interface parser

        A command line argument ``--FLAG`` where FLAG=``self.name`` is added
        which appends ``self.activate`` to the list of extensions. As help
        text the docstring of the extension class is used.
        In most cases this method does not need to be overwritten.

        Args:
            parser: current parser object
        """
        help = self.__doc__[0].lower() + self.__doc__[1:]
        print(parser)

        parser.add_argument(
            self.flag, help=help, nargs=0, dest="extensions", action=IncludeExtensions
        )
        return self

    def activate(self, actions):
        actions = self.register(actions, add_pytorch_project, after="define_structure")
        actions = self.register(
            actions, add_pytorch_structure, after="add_pytorch_project"
        )
        actions = self.register(actions, replace_readme, after="add_pytorch_structure")
        return actions


# TODO : Docstrings
# TODO : Move to init
def gen_src_file_from_template(
    struct, opts, src_path, subpackage, base_name, extension="py", is_source=True
):
    model_path = src_path + [subpackage, "{}.{}".format(base_name, extension)]
    if is_source:
        model_py = templates.src_template(
            opts, subpackage, "{}_{}".format(base_name, extension)
        )
    else:
        model_py = templates.project_root_template(
            opts, subpackage, "{}_{}".format(base_name, extension)
        )
    struct = helpers.ensure(struct, model_path, model_py, helpers.NO_OVERWRITE)
    return struct


# # TODO : Docstrings
# # TODO : Move to init
# def gen_project_file_from_template(
#         struct, opts, src_path, subpackage, base_name, extension="py"
# ):
#     model_path = src_path + [subpackage, "{}.{}".format(base_name, extension)]
#     struct = helpers.ensure(struct, model_path, model_py, helpers.NO_OVERWRITE)
#     return struct


def add_pytorch_structure(struct, opts):
    """Adds basic module for custom extension

    Args:
        struct (dict): project representation as (possibly) nested
            :obj:`dict`.
        opts (dict): given options, see :obj:`create_project` for
            an extensive list.

    Returns:
        struct, opts: updated project representation and options
    """

    src_path = templates.get_src_folder(opts)

    # Base
    struct = gen_src_file_from_template(struct, opts, src_path, "base", "__init__")
    struct = gen_src_file_from_template(
        struct, opts, src_path, "base", "base_data_loader"
    )
    struct = gen_src_file_from_template(struct, opts, src_path, "base", "base_model")
    struct = gen_src_file_from_template(struct, opts, src_path, "base", "base_trainer")

    # Data Loader
    struct = gen_src_file_from_template(
        struct, opts, src_path, "data_loader", "__init__"
    )
    struct = gen_src_file_from_template(
        struct, opts, src_path, "data_loader", "data_loaders"
    )

    # Logger
    struct = gen_src_file_from_template(struct, opts, src_path, "logger", "__init__")
    struct = gen_src_file_from_template(struct, opts, src_path, "logger", "logger")
    struct = gen_src_file_from_template(
        struct, opts, src_path, "logger", "visualization"
    )

    # Model
    struct = gen_src_file_from_template(struct, opts, src_path, "model", "__init__")
    struct = gen_src_file_from_template(struct, opts, src_path, "model", "loss")
    struct = gen_src_file_from_template(struct, opts, src_path, "model", "metric")
    struct = gen_src_file_from_template(struct, opts, src_path, "model", "model")

    # Trainer
    struct = gen_src_file_from_template(struct, opts, src_path, "trainer", "__init__")
    struct = gen_src_file_from_template(struct, opts, src_path, "trainer", "trainer")

    # Utils
    struct = gen_src_file_from_template(struct, opts, src_path, "utils", "__init__")
    struct = gen_src_file_from_template(struct, opts, src_path, "utils", "util")
    struct = gen_src_file_from_template(struct, opts, src_path, "utils", "project")

    # Config
    struct = gen_src_file_from_template(struct, opts, src_path, "", "parse_config")

    # Bin
    struct = gen_src_file_from_template(
        struct, opts, [opts["project"]], "bin", "train", is_source=False
    )
    struct = gen_src_file_from_template(
        struct, opts, [opts["project"]], "bin", "test", is_source=False
    )

    # Config
    struct = gen_src_file_from_template(
        struct,
        opts,
        [opts["project"]],
        "configs",
        "logger_config",
        extension="json",
        is_source=False,
    )

    return struct, opts


def add_pytorch_project(struct, opts):
    """Adds basic module for custom extension

    Args:
        struct (dict): project representation as (possibly) nested
            :obj:`dict`.
        opts (dict): given options, see :obj:`create_project` for
            an extensive list.

    Returns:
        struct, opts: updated project representation and options
    """
    gitignore_all = templates.gitignore_all(opts)

    path = [opts["project"], "data", ".gitignore"]
    struct = helpers.ensure(
        struct, path, templates.gitignore_data(opts), helpers.NO_OVERWRITE
    )
    for folder in ("datasets", "sample", "output"):
        path = [opts["project"], "data", folder, ".gitignore"]
        struct = helpers.ensure(struct, path, gitignore_all, helpers.NO_OVERWRITE)

    path = [opts["project"], "weights", ".gitignore"]
    struct = helpers.ensure(
        struct, path, templates.gitignore_data(opts), helpers.NO_OVERWRITE
    )

    path = [opts["project"], "references", ".gitignore"]
    struct = helpers.ensure(struct, path, "", helpers.NO_OVERWRITE)

    path = [opts["project"], "reports", "figures", ".gitignore"]
    struct = helpers.ensure(struct, path, "", helpers.NO_OVERWRITE)

    path = [opts["project"], "environment.yaml"]
    environment_yaml = templates.environment_yaml(opts)
    struct = helpers.ensure(struct, path, environment_yaml, helpers.NO_OVERWRITE)

    path = [opts["project"], "requirements.txt"]
    struct = helpers.reject(struct, path)

    path = [opts["project"], "configs", ".gitignore"]
    struct = helpers.ensure(struct, path, "", helpers.NO_OVERWRITE)
    return struct, opts


def replace_readme(struct, opts):
    """Replace the readme.md of the markdown extension by our own

    Args:
        struct (dict): project representation as (possibly) nested
            :obj:`dict`.
        opts (dict): given options, see :obj:`create_project` for
            an extensive list.

    Returns:
        struct, opts: updated project representation and options
    """
    # let the markdown extension do its job first
    struct, opts = MarkDown("markdown").markdown(struct, opts)

    file_path = [opts["project"], "README.md"]
    struct = helpers.reject(struct, file_path)
    readme = templates.readme_md(opts)
    struct = helpers.ensure(struct, file_path, readme, helpers.NO_OVERWRITE)
    return struct, opts
