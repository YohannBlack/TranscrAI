# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transcription.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13transcription.proto\x12\rtranscription\"L\n\x14TranscriptionRequest\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\x13\n\x0bisLongAudio\x18\x02 \x01(\x08\x12\x10\n\x08language\x18\x03 \x01(\t\"%\n\x15TranscriptionResponse\x12\x0c\n\x04text\x18\x01 \x01(\t2\xcb\x01\n\nTranscribe\x12W\n\nTranscribe\x12#.transcription.TranscriptionRequest\x1a$.transcription.TranscriptionResponse\x12\x64\n\x13TranscribeStreaming\x12#.transcription.TranscriptionRequest\x1a$.transcription.TranscriptionResponse(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'transcription_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_TRANSCRIPTIONREQUEST']._serialized_start=38
  _globals['_TRANSCRIPTIONREQUEST']._serialized_end=114
  _globals['_TRANSCRIPTIONRESPONSE']._serialized_start=116
  _globals['_TRANSCRIPTIONRESPONSE']._serialized_end=153
  _globals['_TRANSCRIBE']._serialized_start=156
  _globals['_TRANSCRIBE']._serialized_end=359
# @@protoc_insertion_point(module_scope)
