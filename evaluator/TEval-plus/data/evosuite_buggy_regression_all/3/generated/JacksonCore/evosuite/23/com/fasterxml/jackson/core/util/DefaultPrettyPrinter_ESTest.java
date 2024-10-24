/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:33:51 GMT 2023
 */

package com.fasterxml.jackson.core.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.PrettyPrinter;
import com.fasterxml.jackson.core.SerializableString;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.filter.FilteringGeneratorDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.io.SerializedString;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import java.io.BufferedOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
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
      SerializedString serializedString0 = PrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter(serializedString0);
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
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("");
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
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate((JsonGenerator) null, false);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.beforeObjectEntries(jsonGeneratorDelegate0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.JsonGeneratorDelegate", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("Lfl';l3lj~Tka?<M");
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
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((SerializableString) null);
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
      defaultPrettyPrinter0.indentObjectsWith((DefaultPrettyPrinter.Indenter) null);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndObject((JsonGenerator) null, 0);
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
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = new DefaultPrettyPrinter.NopIndenter();
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringGeneratorDelegate filteringGeneratorDelegate0 = new FilteringGeneratorDelegate((JsonGenerator) null, tokenFilter0, true, true);
      defaultPrettyPrinter_NopIndenter0.writeIndentation(filteringGeneratorDelegate0, 1);
      assertEquals(0, filteringGeneratorDelegate0.getMatchCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = new DefaultPrettyPrinter.FixedSpaceIndenter();
      assertTrue(defaultPrettyPrinter_FixedSpaceIndenter0.isInline());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(32);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 65599, (ObjectCodec) null, byteArrayBuilder0);
      defaultPrettyPrinter0.writeStartArray(uTF8JsonGenerator0);
      assertFalse(uTF8JsonGenerator0.canWriteObjectId());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((String) null);
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator((String) null);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator((SerializableString) defaultPrettyPrinter0.DEFAULT_ROOT_VALUE_SEPARATOR);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator((SerializableString) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      SerializedString serializedString0 = PrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator((SerializableString) serializedString0);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator("&");
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      defaultPrettyPrinter0.indentArraysWith((DefaultPrettyPrinter.Indenter) null);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = new DefaultPrettyPrinter.NopIndenter();
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withArrayIndenter(defaultPrettyPrinter_NopIndenter0);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withArrayIndenter((DefaultPrettyPrinter.Indenter) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((String) null);
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = DefaultPrettyPrinter.NopIndenter.instance;
      defaultPrettyPrinter0.indentArraysWith(defaultPrettyPrinter_NopIndenter0);
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withArrayIndenter(defaultPrettyPrinter_NopIndenter0);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter((DefaultPrettyPrinter.Indenter) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = DefaultPrettyPrinter.NopIndenter.instance;
      defaultPrettyPrinter0.indentObjectsWith(defaultPrettyPrinter_NopIndenter0);
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter(defaultPrettyPrinter_NopIndenter0);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(32);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayBuilder0, 2);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, (-879), (ObjectCodec) null, bufferedOutputStream0);
      DefaultPrettyPrinter defaultPrettyPrinter1 = new DefaultPrettyPrinter((SerializableString) null);
      defaultPrettyPrinter1.writeRootValueSeparator(uTF8JsonGenerator0);
      assertEquals(0, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(32);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 65599, (ObjectCodec) null, byteArrayBuilder0);
      defaultPrettyPrinter0.writeRootValueSeparator(uTF8JsonGenerator0);
      assertEquals(1, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(32);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 65599, (ObjectCodec) null, byteArrayBuilder0);
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = DefaultPrettyPrinter.FixedSpaceIndenter.instance;
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter(defaultPrettyPrinter_FixedSpaceIndenter0);
      defaultPrettyPrinter1.writeStartObject(uTF8JsonGenerator0);
      assertTrue(defaultPrettyPrinter_FixedSpaceIndenter0.isInline());
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(0);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayBuilder0, 46);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, (ObjectCodec) null, bufferedOutputStream0);
      defaultPrettyPrinter0.writeStartObject(uTF8JsonGenerator0);
      assertTrue(uTF8JsonGenerator0.canOmitFields());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = new DefaultPrettyPrinter(defaultPrettyPrinter0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      defaultPrettyPrinter1._spacesInObjectEntries = false;
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(37);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayBuilder0, 2);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1097, (ObjectCodec) null, bufferedOutputStream0);
      defaultPrettyPrinter1.writeObjectFieldValueSeparator(uTF8JsonGenerator0);
      assertEquals(1, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(0);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayBuilder0, 46);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, (ObjectCodec) null, bufferedOutputStream0);
      defaultPrettyPrinter0.writeObjectFieldValueSeparator(uTF8JsonGenerator0);
      assertEquals(3, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndObject((JsonGenerator) null, (-2863));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndObject((JsonGenerator) null, 3);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultIndenter", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      DefaultIndenter defaultIndenter0 = DefaultIndenter.SYSTEM_LINEFEED_INSTANCE;
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(0);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayBuilder0, 46);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, (ObjectCodec) null, bufferedOutputStream0);
      defaultPrettyPrinter0.indentArraysWith(defaultIndenter0);
      defaultPrettyPrinter0.writeStartArray(uTF8JsonGenerator0);
      assertEquals(55296, GeneratorBase.SURR1_FIRST);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, defaultPrettyPrinter0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(32);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayBuilder0, 2);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, (-879), (ObjectCodec) null, bufferedOutputStream0);
      defaultPrettyPrinter0.writeEndArray(uTF8JsonGenerator0, 114);
      assertEquals(56319, GeneratorBase.SURR1_LAST);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultIndenter defaultIndenter0 = DefaultIndenter.SYSTEM_LINEFEED_INSTANCE;
      defaultPrettyPrinter0.indentArraysWith(defaultIndenter0);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndArray((JsonGenerator) null, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }
}
