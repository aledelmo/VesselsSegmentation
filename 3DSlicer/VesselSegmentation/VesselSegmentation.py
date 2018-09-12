#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import unittest
import ctk
import qt
import slicer
import sys
import pickle
import vtk
from builtins import int, range
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import processing_vs as vs

__author__ = 'Alessandro Delmonte'
__email__ = 'delmonte.ale92@gmail.com'


class VesselSegmentation:
    def __init__(self, parent):
        parent.title = 'Vessels'
        parent.categories = ['IMAG2', 'Pelvic Segmentation']
        parent.dependencies = []
        parent.contributors = ['Alessandro Delmonte (IMAG2)']
        parent.helpText = '''INSERT HELP TEXT.'''
        parent.acknowledgementText = '''Module developed for 3DSlicer (<a>http://www.slicer.org</a>)'''

        self.parent = parent

        module_dir = os.path.dirname(self.parent.path)
        icon_path = os.path.join(module_dir, 'Resources', 'icon.png')
        if os.path.isfile(icon_path):
            parent.icon = qt.QIcon(icon_path)

        try:
            slicer.selfTests
        except AttributeError:
            slicer.selfTests = {}
        slicer.selfTests['VesselSegmentation'] = self.runTest

    def __repr__(self):
        return 'VesselSegmentation(parent={})'.format(self.parent)

    def __str__(self):
        return 'VesselSegmentation module initialization class.'

    @staticmethod
    def runTest():
        tester = VesselSegmentationTest()
        tester.runTest()


class VesselSegmentationWidget:
    def __init__(self, parent=None):
        self.moduleName = self.__class__.__name__
        if self.moduleName.endswith('Widget'):
            self.moduleName = self.moduleName[:-6]
        settings = qt.QSettings()
        try:
            self.developerMode = settings.value('Developer/DeveloperMode').lower() == 'true'
        except AttributeError:
            self.developerMode = settings.value('Developer/DeveloperMode') is True

        self.tmp = tempfile.mkdtemp()
        self.logic = VesselSegmentationLogic(self.tmp)

        if not parent:
            self.parent = slicer.qMRMLWidget()
            self.parent.setLayout(qt.QVBoxLayout())
            self.parent.setMRMLScene(slicer.mrmlScene)
        else:
            self.parent = parent
        self.layout = self.parent.layout()

        if not parent:
            self.setup()
            self.parent.show()

    def __repr__(self):
        return 'VesselSegmentationsWidget(parent={})'.format(self.parent)

    def __str__(self):
        return 'VesselSegmentation GUI class'

    def setup(self):
        vs_collapsible_button = ctk.ctkCollapsibleButton()
        vs_collapsible_button.text = 'Vessels Segmentation'

        self.layout.addWidget(vs_collapsible_button)

        vs_form_layout = qt.QFormLayout(vs_collapsible_button)
        vs_form_layout.setVerticalSpacing(13)

        self.volume_selector = slicer.qMRMLNodeComboBox()
        self.volume_selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.volume_selector.selectNodeUponCreation = True
        self.volume_selector.addEnabled = False
        self.volume_selector.removeEnabled = False
        self.volume_selector.noneEnabled = False
        self.volume_selector.showHidden = False
        self.volume_selector.renameEnabled = False
        self.volume_selector.showChildNodeTypes = False
        self.volume_selector.setMRMLScene(slicer.mrmlScene)

        self.volume_selector.connect('nodeActivated(vtkMRMLNode*)', self.on_volume_select)

        self.volume_node = self.volume_selector.currentNode()

        vs_form_layout.addRow('Reference Volume: ', self.volume_selector)

        self.output_selector = slicer.qMRMLNodeComboBox()
        self.output_selector.nodeTypes = ['vtkMRMLLabelMapVolumeNode']
        self.output_selector.selectNodeUponCreation = True
        self.output_selector.addEnabled = True
        self.output_selector.removeEnabled = False
        self.output_selector.noneEnabled = False
        self.output_selector.showHidden = False
        self.output_selector.renameEnabled = True
        self.output_selector.showChildNodeTypes = False
        self.output_selector.setMRMLScene(slicer.mrmlScene)

        self.output_selector.connect('currentNodeChanged(vtkMRMLNode*)', self.on_output_select)

        self.output_node = self.output_selector.currentNode()

        vs_form_layout.addRow('Output Seeds Label: ', self.output_selector)

        self.compute_button = qt.QPushButton('Compute')
        self.compute_button.enabled = True

        self.compute_button.connect('clicked(bool)', self.on_compute_button)
        vs_form_layout.addRow(self.compute_button)

        line = qt.QFrame()
        line.setFrameShape(qt.QFrame().HLine)
        line.setFrameShadow(qt.QFrame().Sunken)
        line.setStyleSheet("min-height: 24px")
        vs_form_layout.addRow(line)

        options_grid_layout = qt.QGridLayout()
        options_grid_layout.setColumnStretch(0, 1)
        options_grid_layout.setColumnStretch(1, 1)
        options_grid_layout.setColumnStretch(2, 1)

        type_text_label = qt.QLabel()
        type_text_label.setText('Vessel type: ')
        options_grid_layout.addWidget(type_text_label, 0, 0, 0)

        self.radio_art = qt.QRadioButton('Arteries')
        self.radio_art.setChecked(True)
        self.radio_vein = qt.QRadioButton('Veins')
        options_grid_layout.addWidget(self.radio_art, 0, 1, 0)
        options_grid_layout.addWidget(self.radio_vein, 0, 2, 0)

        vs_form_layout.addRow(options_grid_layout)

        groupbox = qt.QGroupBox()
        groupbox.setTitle('Choose one or more scalar map:')
        options_grid_layout = qt.QGridLayout(groupbox)
        options_grid_layout.setColumnStretch(0, 0)
        options_grid_layout.setColumnStretch(0, 1)
        options_grid_layout.setColumnStretch(0, 2)

        self.markups_selector = slicer.qSlicerSimpleMarkupsWidget()
        self.markups_selector.objectName = 'seedFiducialsNodeSelector'
        self.markups_selector = slicer.qSlicerSimpleMarkupsWidget()
        self.markups_selector.objectName = 'seedFiducialsNodeSelector'
        self.markups_selector.toolTip = "Select the fiducials to use as the origin of the algorithm."
        self.markups_selector.setNodeBaseName("OriginSeeds")
        self.markups_selector.defaultNodeColor = qt.QColor(202, 169, 250)
        self.markups_selector.maximumHeight = 250
        self.markups_selector.markupsSelectorComboBox().noneEnabled = False
        vs_form_layout.addRow("Initial points:", self.markups_selector)
        self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                            self.markups_selector, 'setMRMLScene(vtkMRMLScene*)')

        self.layout.addStretch(1)

        if self.developerMode:

            def createHLayout(elements):
                widget = qt.QWidget()
                rowLayout = qt.QHBoxLayout()
                widget.setLayout(rowLayout)
                for element in elements:
                    rowLayout.addWidget(element)
                return widget

            """Developer interface"""
            self.reloadCollapsibleButton = ctk.ctkCollapsibleButton()
            self.reloadCollapsibleButton.text = "Reload && Test"
            self.layout.addWidget(self.reloadCollapsibleButton)
            reloadFormLayout = qt.QFormLayout(self.reloadCollapsibleButton)

            self.reloadButton = qt.QPushButton("Reload")
            self.reloadButton.toolTip = "Reload this module."
            self.reloadButton.name = "ScriptedLoadableModuleTemplate Reload"
            self.reloadButton.connect('clicked()', self.onReload)

            self.reloadAndTestButton = qt.QPushButton("Reload and Test")
            self.reloadAndTestButton.toolTip = "Reload this module and then run the self tests."
            self.reloadAndTestButton.connect('clicked()', self.onReloadAndTest)

            self.editSourceButton = qt.QPushButton("Edit")
            self.editSourceButton.toolTip = "Edit the module's source code."
            self.editSourceButton.connect('clicked()', self.onEditSource)

            self.restartButton = qt.QPushButton("Restart Slicer")
            self.restartButton.toolTip = "Restart Slicer"
            self.restartButton.name = "ScriptedLoadableModuleTemplate Restart"
            self.restartButton.connect('clicked()', slicer.app.restart)

            reloadFormLayout.addWidget(
                createHLayout([self.reloadButton, self.reloadAndTestButton, self.editSourceButton, self.restartButton]))

    def on_volume_select(self):
        self.volume_node = self.volume_selector.currentNode()

    def on_output_select(self):
        self.output_node = self.output_selector.currentNode()

    def on_compute_button(self):

        if self.output_node and self.volume_node and self.markups_selector.currentNode():
            current_seeds_node = self.markups_selector.currentNode()
            fid_list = []
            for n in range(current_seeds_node.GetNumberOfFiducials()):
                current = [0, 0, 0]
                current_seeds_node.GetNthFiducialPosition(n, current)
                fid_list.append(current)

            properties = {'useCompression': 0}
            volume_path = os.path.join(self.tmp, 'reference.nii')
            slicer.util.saveNode(self.volume_node, volume_path, properties)

            if self.radio_art.isChecked():
                vessel_type = 0
            else:
                vessel_type = 1

            segmentation_path = self.logic.segment(volume_path, fid_list, vessel_type)

            _, new_node = slicer.util.loadVolume(segmentation_path, returnNode=True)

            node_name = self.output_node.GetName()
            slicer.mrmlScene.RemoveNode(self.output_node)
            new_node.SetName(node_name)

            self.cleanup()

    def onReload(self):

        print('\n' * 2)
        print('-' * 30)
        print('Reloading module: ' + self.moduleName)
        print('-' * 30)
        print('\n' * 2)

        slicer.util.reloadScriptedModule(self.moduleName)

    def onReloadAndTest(self):
        try:
            self.onReload()
            test = slicer.selfTests[self.moduleName]
            test()
        except Exception:
            import traceback
            traceback.print_exc()
            errorMessage = "Reload and Test: Exception!\n\n" + "See Python Console for Stack Trace"
            slicer.util.errorDisplay(errorMessage)

    def onEditSource(self):
        filePath = slicer.util.modulePath(self.moduleName)
        qt.QDesktopServices.openUrl(qt.QUrl("file:///" + filePath, qt.QUrl.TolerantMode))

    def cleanup(self):
        for filename in os.listdir(self.tmp):
            path = os.path.join(self.tmp, filename)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)


class VesselSegmentationLogic:
    def __init__(self, temp_folder):
        self.temp_folder = temp_folder

    def segment(self, reference_path, markups_list, vessel_type):
        output_path = os.path.join(self.temp_folder, 'output-label.nii')
        markups_path = os.path.join(self.temp_folder, 'init.pkl')

        with open(markups_path, 'wb') as handle:
            pickle.dump(markups_list, handle)

        with open('/Users/imag2/Desktop/VesselsSegmentation/test_dataset/init.pkl', 'wb') as handle:
            pickle.dump(markups_list, handle)

        if vessel_type == 1:
            vessel = 'vein'
        else:
            vessel = 'artery'

        vs.proc(reference_path, markups_path, vessel, output_path, True)

        return output_path


class VesselSegmentationTest(unittest.TestCase):

    def __init__(self):
        super(VesselSegmentationTest, self).__init__()

    def __repr__(self):
        return 'VesselSegmentation(). Derived from {}'.format(unittest.TestCase)

    def __str__(self):
        return 'VesselSegmentation test class'

    def runTest(self, scenario=None):
        pass
