/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:55:40 GMT 2023
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
import java.io.StringWriter;
import java.io.Writer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonGeneratorImpl_ESTest extends JsonGeneratorImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 0, (ObjectCodec) null, (Writer) null);
      writerBasedJsonGenerator0.version();
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, (Writer) null);
      writerBasedJsonGenerator0.getCharacterEscapes();
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      MockFile mockFile0 = new MockFile("6G6", "6G6");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1023, (ObjectCodec) null, mockFileOutputStream0);
      int int0 = uTF8JsonGenerator0.getHighestEscapedChar();
      assertEquals(127, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, (Writer) null);
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.setRootValueSeparator((SerializableString) null);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, (Writer) null);
      writerBasedJsonGenerator0.writeStringField("=kUT4l^u0kjhmt", "=kUT4l^u0kjhmt");
      assertEquals(30, writerBasedJsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 3, (ObjectCodec) null, (Writer) null);
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.WRITE_BIGDECIMAL_AS_PLAIN;
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.enable(jsonGenerator_Feature0);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter(1);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 3, (ObjectCodec) null, stringWriter0);
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.QUOTE_FIELD_NAMES;
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.configure(jsonGenerator_Feature0, true);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, (ObjectCodec) null, (Writer) null);
      writerBasedJsonGenerator0._checkStdFeatureChanges(2702, 1);
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, (ObjectCodec) null, (Writer) null);
      writerBasedJsonGenerator0._checkStdFeatureChanges(3, 1);
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 0, (ObjectCodec) null, (Writer) null);
      writerBasedJsonGenerator0.setHighestNonEscapedChar((-4089));
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter(1);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 3, (ObjectCodec) null, stringWriter0);
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.setCharacterEscapes((CharacterEscapes) null);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }
}
