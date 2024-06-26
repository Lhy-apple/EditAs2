/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:35:22 GMT 2023
 */

package com.fasterxml.jackson.core.json;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.SerializableString;
import com.fasterxml.jackson.core.io.CharacterEscapes;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import java.io.OutputStream;
import java.io.StringWriter;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonGeneratorImpl_ESTest extends JsonGeneratorImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, (ObjectCodec) null, stringWriter0);
      writerBasedJsonGenerator0.version();
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter(2);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, (ObjectCodec) null, stringWriter0);
      writerBasedJsonGenerator0.getCharacterEscapes();
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 0, (ObjectCodec) null, (OutputStream) null);
      int int0 = uTF8JsonGenerator0.getHighestEscapedChar();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, (ObjectCodec) null, stringWriter0);
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.setRootValueSeparator((SerializableString) null);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, stringWriter0);
      writerBasedJsonGenerator0.writeStringField("Gm#", "-1");
      assertEquals(7, writerBasedJsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter(0);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, (-2620), (ObjectCodec) null, stringWriter0);
      assertEquals(127, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, (-660), (ObjectCodec) null, stringWriter0);
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(3);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, (ObjectCodec) null, byteArrayBuilder0, byteArrayBuilder0.NO_BYTES, 2, false);
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.IGNORE_UNKNOWN;
      JsonGenerator jsonGenerator0 = uTF8JsonGenerator0.enable(jsonGenerator_Feature0);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 3, (ObjectCodec) null, stringWriter0);
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.QUOTE_FIELD_NAMES;
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.enable(jsonGenerator_Feature0);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 0, (ObjectCodec) null, stringWriter0);
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.overrideStdFeatures(15, 15);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, (ObjectCodec) null, stringWriter0);
      writerBasedJsonGenerator0._checkStdFeatureChanges(3, 3);
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, (ObjectCodec) null, stringWriter0);
      writerBasedJsonGenerator0.setHighestNonEscapedChar(3);
      assertEquals(3, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 3, (ObjectCodec) null, stringWriter0);
      writerBasedJsonGenerator0.setHighestNonEscapedChar((-3641));
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 3, (ObjectCodec) null, stringWriter0);
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.setCharacterEscapes((CharacterEscapes) null);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }
}
