/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:18:39 GMT 2023
 */

package com.fasterxml.jackson.core.json;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
import java.io.Writer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonGeneratorImpl_ESTest extends JsonGeneratorImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, (Writer) null);
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
      Object object0 = new Object();
      IOContext iOContext0 = new IOContext(bufferRecycler0, object0, false);
      byte[] byteArray0 = new byte[6];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, (ObjectCodec) null, (OutputStream) null, byteArray0, 23, false);
      int int0 = uTF8JsonGenerator0.getHighestEscapedChar();
      assertEquals(0, int0);
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
      writerBasedJsonGenerator0.writeStringField("", "");
      assertEquals(2, writerBasedJsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(1114111);
      UTF8JsonGenerator uTF8JsonGenerator0 = null;
      try {
        uTF8JsonGenerator0 = new UTF8JsonGenerator((IOContext) null, 1114111, (ObjectCodec) null, byteArrayBuilder0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.json.UTF8JsonGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, (Writer) null);
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
      
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.ESCAPE_NON_ASCII;
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.enable(jsonGenerator_Feature0);
      assertEquals(127, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, (Writer) null);
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.QUOTE_FIELD_NAMES;
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.configure(jsonGenerator_Feature0, true);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, stringWriter0);
      writerBasedJsonGenerator0._checkStdFeatureChanges(362, 362);
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, stringWriter0);
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.overrideStdFeatures(3, 3);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 0, (ObjectCodec) null, (Writer) null);
      writerBasedJsonGenerator0.setHighestNonEscapedChar((-3319));
      assertEquals(0, writerBasedJsonGenerator0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 1, (ObjectCodec) null, (Writer) null);
      JsonGenerator jsonGenerator0 = writerBasedJsonGenerator0.setCharacterEscapes((CharacterEscapes) null);
      assertEquals(0, jsonGenerator0.getHighestEscapedChar());
  }
}