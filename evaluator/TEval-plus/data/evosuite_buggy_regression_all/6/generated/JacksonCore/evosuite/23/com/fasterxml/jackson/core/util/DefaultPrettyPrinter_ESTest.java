/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:18:10 GMT 2023
 */

package com.fasterxml.jackson.core.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.SerializableString;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultPrettyPrinter_ESTest extends DefaultPrettyPrinter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.createInstance();
      assertFalse(defaultPrettyPrinter1.equals((Object)defaultPrettyPrinter0));
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeObjectEntrySeparator((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.beforeArrayValues((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter$FixedSpaceIndenter", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.beforeObjectEntries((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultIndenter", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withoutSpacesInObjectEntries();
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withSpacesInObjectEntries();
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeArrayValueSeparator((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = new DefaultPrettyPrinter.NopIndenter();
      defaultPrettyPrinter0.indentArraysWith(defaultPrettyPrinter_NopIndenter0);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeStartArray((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = DefaultPrettyPrinter.NopIndenter.instance;
      defaultPrettyPrinter_NopIndenter0.writeIndentation((JsonGenerator) null, (-681));
      assertTrue(defaultPrettyPrinter_NopIndenter0.isInline());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = new DefaultPrettyPrinter.FixedSpaceIndenter();
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      defaultPrettyPrinter0._objectIndenter = (DefaultPrettyPrinter.Indenter) defaultPrettyPrinter_FixedSpaceIndenter0;
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter(defaultPrettyPrinter_FixedSpaceIndenter0);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, (ObjectCodec) null, byteArrayBuilder0);
      defaultPrettyPrinter0.writeEndArray(uTF8JsonGenerator0, (-447));
      assertFalse(uTF8JsonGenerator0.canWriteFormattedNumbers());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("CSOj ");
      defaultPrettyPrinter0.indentArraysWith((DefaultPrettyPrinter.Indenter) null);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((String) null);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator((SerializableString) defaultPrettyPrinter0.DEFAULT_ROOT_VALUE_SEPARATOR);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("");
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator("");
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator("expected padding character '");
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultIndenter defaultIndenter0 = DefaultIndenter.SYSTEM_LINEFEED_INSTANCE;
      defaultPrettyPrinter0.indentObjectsWith(defaultIndenter0);
      assertFalse(defaultIndenter0.isInline());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("OY_l(]");
      defaultPrettyPrinter0.indentObjectsWith((DefaultPrettyPrinter.Indenter) null);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = DefaultPrettyPrinter.FixedSpaceIndenter.instance;
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withArrayIndenter(defaultPrettyPrinter_FixedSpaceIndenter0);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("CSOj ");
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withArrayIndenter((DefaultPrettyPrinter.Indenter) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = DefaultPrettyPrinter.NopIndenter.instance;
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("6/0>v");
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter(defaultPrettyPrinter_NopIndenter0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate((JsonGenerator) null, true);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter1.writeEndObject(jsonGeneratorDelegate0, (byte) (-116));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.JsonGeneratorDelegate", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter((DefaultPrettyPrinter.Indenter) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator((String) null);
      MockFile mockFile0 = new MockFile("com.fasterxml.jackson.core.util.DefaultPrettyPrinter$FixedSpaceIndenter", "com.fasterxml.jackson.core.util.DefaultPrettyPrinter$FixedSpaceIndenter");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 579, (ObjectCodec) null, mockFileOutputStream0);
      defaultPrettyPrinter1.writeRootValueSeparator(uTF8JsonGenerator0);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
      assertEquals(0, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, (ObjectCodec) null, byteArrayBuilder0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(uTF8JsonGenerator0);
      defaultPrettyPrinter0.writeRootValueSeparator(jsonGeneratorDelegate0);
      assertEquals(1, jsonGeneratorDelegate0.getOutputBuffered());
      assertEquals(1, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("CSOj ");
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, false);
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = new DefaultPrettyPrinter.FixedSpaceIndenter();
      defaultPrettyPrinter0._objectIndenter = (DefaultPrettyPrinter.Indenter) defaultPrettyPrinter_FixedSpaceIndenter0;
      MockFile mockFile0 = new MockFile("", "com.fasterxml.jackson.core.util.DefaultPrettyPrinter$FixedSpaceIndenter");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, (ObjectCodec) null, mockFileOutputStream0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(uTF8JsonGenerator0, false);
      defaultPrettyPrinter0.writeStartObject(jsonGeneratorDelegate0);
      assertEquals(0, jsonGeneratorDelegate0.getHighestEscapedChar());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("CSOj ");
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      MockFile mockFile0 = new MockFile("", "com.fasterxml.jackson.core.util.DefaultPrettyPrinter$FixedSpaceIndenter");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, (ObjectCodec) null, mockFileOutputStream0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(uTF8JsonGenerator0, true);
      defaultPrettyPrinter0.writeStartObject(jsonGeneratorDelegate0);
      assertFalse(jsonGeneratorDelegate0.canWriteTypeId());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      defaultPrettyPrinter0._spacesInObjectEntries = false;
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeObjectFieldValueSeparator((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, (ObjectCodec) null, byteArrayBuilder0);
      defaultPrettyPrinter0.writeObjectFieldValueSeparator(uTF8JsonGenerator0);
      assertEquals(3, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, (ObjectCodec) null, byteArrayBuilder0);
      defaultPrettyPrinter0.writeEndObject(uTF8JsonGenerator0, 1952);
      assertEquals(2, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("T");
      DefaultIndenter defaultIndenter0 = new DefaultIndenter();
      defaultPrettyPrinter0._arrayIndenter = (DefaultPrettyPrinter.Indenter) defaultIndenter0;
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeStartArray((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultIndenter defaultIndenter0 = new DefaultIndenter();
      defaultPrettyPrinter0._arrayIndenter = (DefaultPrettyPrinter.Indenter) defaultIndenter0;
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndArray((JsonGenerator) null, 152);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultIndenter", e);
      }
  }
}