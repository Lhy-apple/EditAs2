/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:35:44 GMT 2023
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
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import java.io.IOException;
import java.io.PipedOutputStream;
import java.io.StringWriter;
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
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 65599, (ObjectCodec) null, stringWriter0);
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      defaultPrettyPrinter0.writeObjectEntrySeparator(writerBasedJsonGenerator0);
      assertEquals(2, writerBasedJsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("y~[uIT7yB_e{C");
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate((JsonGenerator) null);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.beforeArrayValues(jsonGeneratorDelegate0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.JsonGeneratorDelegate", e);
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
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withSpacesInObjectEntries();
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      BufferRecycler bufferRecycler0 = new BufferRecycler(0, 115);
      Object object0 = new Object();
      IOContext iOContext0 = new IOContext(bufferRecycler0, object0, true);
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      byte[] byteArray0 = new byte[6];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 3, (ObjectCodec) null, pipedOutputStream0, byteArray0, 32, false);
      try { 
        defaultPrettyPrinter0.writeArrayValueSeparator(uTF8JsonGenerator0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((String) null);
      DefaultPrettyPrinter.NopIndenter defaultPrettyPrinter_NopIndenter0 = DefaultPrettyPrinter.NopIndenter.instance;
      defaultPrettyPrinter0.indentArraysWith(defaultPrettyPrinter_NopIndenter0);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndArray((JsonGenerator) null, 55296);
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
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = new DefaultPrettyPrinter.FixedSpaceIndenter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 65599, (ObjectCodec) null, stringWriter0);
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      defaultPrettyPrinter0.indentObjectsWith(defaultPrettyPrinter_FixedSpaceIndenter0);
      defaultPrettyPrinter0.writeStartObject(writerBasedJsonGenerator0);
      assertEquals(0, writerBasedJsonGenerator0.getFormatFeatures());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
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
  public void test09()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("<baFxq4");
      DefaultPrettyPrinter defaultPrettyPrinter1 = new DefaultPrettyPrinter(defaultPrettyPrinter0, (SerializableString) null);
      DefaultPrettyPrinter defaultPrettyPrinter2 = defaultPrettyPrinter1.withRootSeparator((SerializableString) null);
      assertSame(defaultPrettyPrinter2, defaultPrettyPrinter1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("<baFxq4");
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator("\n");
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator((String) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("BtJw:qjFs");
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withRootSeparator("BtJw:qjFs");
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("BtJw:qjFs");
      defaultPrettyPrinter0.indentArraysWith((DefaultPrettyPrinter.Indenter) null);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("");
      defaultPrettyPrinter0.indentObjectsWith((DefaultPrettyPrinter.Indenter) null);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = DefaultPrettyPrinter.FixedSpaceIndenter.instance;
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withArrayIndenter(defaultPrettyPrinter_FixedSpaceIndenter0);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("BtJw:qjFs");
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withArrayIndenter((DefaultPrettyPrinter.Indenter) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("<baFxq4");
      DefaultIndenter defaultIndenter0 = DefaultIndenter.SYSTEM_LINEFEED_INSTANCE;
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter(defaultIndenter0);
      assertSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("com.fasterxml.jackson.core.util.DefaultPrettyPrinter");
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withObjectIndenter((DefaultPrettyPrinter.Indenter) null);
      assertNotSame(defaultPrettyPrinter1, defaultPrettyPrinter0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("<baFxq4");
      DefaultPrettyPrinter defaultPrettyPrinter1 = new DefaultPrettyPrinter(defaultPrettyPrinter0, (SerializableString) null);
      defaultPrettyPrinter1.writeRootValueSeparator((JsonGenerator) null);
      assertFalse(defaultPrettyPrinter0.equals((Object)defaultPrettyPrinter1));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("com.fasterxml.jackson.core.util.DefaultPrettyPrinter");
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeRootValueSeparator((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 65599, (ObjectCodec) null, stringWriter0);
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((String) null);
      defaultPrettyPrinter0.writeStartObject(writerBasedJsonGenerator0);
      assertTrue(writerBasedJsonGenerator0.canWriteFormattedNumbers());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
      DefaultPrettyPrinter defaultPrettyPrinter1 = defaultPrettyPrinter0.withoutSpacesInObjectEntries();
      // Undeclared exception!
      try { 
        defaultPrettyPrinter1.writeObjectFieldValueSeparator((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("ALLOW_MISSING_VALUES");
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
  public void test24()  throws Throwable  {
      DefaultPrettyPrinter.FixedSpaceIndenter defaultPrettyPrinter_FixedSpaceIndenter0 = new DefaultPrettyPrinter.FixedSpaceIndenter();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 65599, (ObjectCodec) null, stringWriter0);
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((String) null);
      defaultPrettyPrinter0.indentObjectsWith(defaultPrettyPrinter_FixedSpaceIndenter0);
      defaultPrettyPrinter0.writeEndObject(writerBasedJsonGenerator0, 55296);
      assertEquals(2, writerBasedJsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter();
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
  public void test26()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("<baFxq4");
      DefaultIndenter defaultIndenter0 = DefaultIndenter.SYSTEM_LINEFEED_INSTANCE;
      defaultPrettyPrinter0.indentArraysWith(defaultIndenter0);
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
  public void test27()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter((SerializableString) null);
      DefaultIndenter defaultIndenter0 = DefaultIndenter.SYSTEM_LINEFEED_INSTANCE;
      defaultPrettyPrinter0.indentArraysWith(defaultIndenter0);
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndArray((JsonGenerator) null, 55296);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultIndenter", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DefaultPrettyPrinter defaultPrettyPrinter0 = new DefaultPrettyPrinter("BtJw:qjFs");
      // Undeclared exception!
      try { 
        defaultPrettyPrinter0.writeEndArray((JsonGenerator) null, (-1115));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.util.DefaultPrettyPrinter", e);
      }
  }
}
