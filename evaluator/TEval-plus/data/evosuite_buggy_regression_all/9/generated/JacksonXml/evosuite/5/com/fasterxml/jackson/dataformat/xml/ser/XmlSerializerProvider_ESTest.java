/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:01:57 GMT 2023
 */

package com.fasterxml.jackson.dataformat.xml.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.ctc.wstx.api.WriterConfig;
import com.ctc.wstx.sw.ISOLatin1XmlWriter;
import com.ctc.wstx.sw.NonNsStreamWriter;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.SerializerFactory;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import com.fasterxml.jackson.databind.util.TokenBuffer;
import com.fasterxml.jackson.dataformat.xml.ser.ToXmlGenerator;
import com.fasterxml.jackson.dataformat.xml.ser.XmlSerializerProvider;
import com.fasterxml.jackson.dataformat.xml.util.XmlRootNameLookup;
import java.io.IOException;
import java.sql.SQLIntegrityConstraintViolationException;
import java.sql.SQLTransientException;
import java.sql.SQLWarning;
import javax.xml.namespace.QName;
import org.codehaus.stax2.XMLStreamWriter2;
import org.codehaus.stax2.util.StreamWriter2Delegate;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlSerializerProvider_ESTest extends XmlSerializerProvider_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      // Undeclared exception!
      try { 
        xmlSerializerProvider0._startRootArray((ToXmlGenerator) null, (QName) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.ser.XmlSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      DefaultSerializerProvider defaultSerializerProvider0 = xmlSerializerProvider0.copy();
      assertNotSame(xmlSerializerProvider0, defaultSerializerProvider0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      SerializationConfig serializationConfig0 = new SerializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      XmlSerializerProvider xmlSerializerProvider1 = new XmlSerializerProvider(xmlSerializerProvider0, serializationConfig0, (SerializerFactory) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      BufferRecycler bufferRecycler0 = jsonFactory0._getBufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      MockPrintStream mockPrintStream0 = new MockPrintStream("JSON");
      WriterConfig writerConfig0 = WriterConfig.createJ2MEDefaults();
      ISOLatin1XmlWriter iSOLatin1XmlWriter0 = new ISOLatin1XmlWriter(mockPrintStream0, writerConfig0, true);
      NonNsStreamWriter nonNsStreamWriter0 = new NonNsStreamWriter(iSOLatin1XmlWriter0, "JSON", writerConfig0);
      ToXmlGenerator toXmlGenerator0 = new ToXmlGenerator(iOContext0, 17, 1, objectMapper0, nonNsStreamWriter0);
      // Undeclared exception!
      try { 
        xmlSerializerProvider1.serializeValue((JsonGenerator) toXmlGenerator0, (Object) mockPrintStream0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.cfg.MapperConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      // Undeclared exception!
      try { 
        xmlSerializerProvider0.createInstance((SerializationConfig) null, (SerializerFactory) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      StreamWriter2Delegate streamWriter2Delegate0 = new StreamWriter2Delegate((XMLStreamWriter2) null);
      ToXmlGenerator toXmlGenerator0 = new ToXmlGenerator((IOContext) null, 3, 1, objectMapper0, streamWriter2Delegate0);
      // Undeclared exception!
      try { 
        xmlSerializerProvider0.serializeValue((JsonGenerator) toXmlGenerator0, (Object) toXmlGenerator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.ser.XmlSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider((XmlRootNameLookup) null);
      // Undeclared exception!
      try { 
        xmlSerializerProvider0.serializeValue((JsonGenerator) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.ser.XmlSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      TokenBuffer tokenBuffer0 = new TokenBuffer(jsonParser0);
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      // Undeclared exception!
      try { 
        xmlSerializerProvider0.serializeValue((JsonGenerator) tokenBuffer0, (Object) "JSON");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      // Undeclared exception!
      try { 
        xmlSerializerProvider0.serializeValue((JsonGenerator) null, (Object) null, javaType0, (JsonSerializer<Object>) beanSerializer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.ser.XmlSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      TokenBuffer tokenBuffer0 = new TokenBuffer(jsonParser0);
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      xmlSerializerProvider0.serializeValue((JsonGenerator) tokenBuffer0, (Object) jsonParser0, javaType0, (JsonSerializer<Object>) beanSerializer0);
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      SerializationConfig serializationConfig0 = new SerializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      XmlSerializerProvider xmlSerializerProvider1 = new XmlSerializerProvider(xmlSerializerProvider0, serializationConfig0, (SerializerFactory) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      BufferRecycler bufferRecycler0 = jsonFactory0._getBufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      MockPrintStream mockPrintStream0 = new MockPrintStream("JSON");
      WriterConfig writerConfig0 = WriterConfig.createJ2MEDefaults();
      ISOLatin1XmlWriter iSOLatin1XmlWriter0 = new ISOLatin1XmlWriter(mockPrintStream0, writerConfig0, true);
      NonNsStreamWriter nonNsStreamWriter0 = new NonNsStreamWriter(iSOLatin1XmlWriter0, "JSON", writerConfig0);
      ToXmlGenerator toXmlGenerator0 = new ToXmlGenerator(iOContext0, 1, 0, objectMapper0, nonNsStreamWriter0);
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.QUOTE_FIELD_NAMES;
      // Undeclared exception!
      try { 
        xmlSerializerProvider1.serializeValue((JsonGenerator) toXmlGenerator0, (Object) jsonGenerator_Feature0, (JavaType) null, xmlSerializerProvider0.DEFAULT_NULL_KEY_SERIALIZER);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.util.XmlRootNameLookup", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      TokenBuffer tokenBuffer0 = new TokenBuffer(jsonParser0);
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      JavaType javaType0 = TypeFactory.unknownType();
      // Undeclared exception!
      try { 
        xmlSerializerProvider0.serializeValue((JsonGenerator) tokenBuffer0, (Object) tokenBuffer0, javaType0, (JsonSerializer<Object>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      SerializationConfig serializationConfig0 = new SerializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      XmlSerializerProvider xmlSerializerProvider1 = new XmlSerializerProvider(xmlSerializerProvider0, serializationConfig0, (SerializerFactory) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      BufferRecycler bufferRecycler0 = jsonFactory0._getBufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      MockPrintStream mockPrintStream0 = new MockPrintStream("JSON");
      WriterConfig writerConfig0 = WriterConfig.createJ2MEDefaults();
      ISOLatin1XmlWriter iSOLatin1XmlWriter0 = new ISOLatin1XmlWriter(mockPrintStream0, writerConfig0, false);
      NonNsStreamWriter nonNsStreamWriter0 = new NonNsStreamWriter(iSOLatin1XmlWriter0, "JSON", writerConfig0);
      ToXmlGenerator toXmlGenerator0 = new ToXmlGenerator(iOContext0, 4, 1, objectMapper0, nonNsStreamWriter0);
      xmlSerializerProvider1._serializeXmlNull(toXmlGenerator0);
      assertFalse(xmlSerializerProvider0.equals((Object)xmlSerializerProvider1));
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      SerializationConfig serializationConfig0 = new SerializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      XmlSerializerProvider xmlSerializerProvider1 = new XmlSerializerProvider(xmlSerializerProvider0, serializationConfig0, (SerializerFactory) null);
      try { 
        xmlSerializerProvider1._serializeXmlNull((JsonGenerator) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // [no message for java.lang.NullPointerException]
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      StreamWriter2Delegate streamWriter2Delegate0 = new StreamWriter2Delegate((XMLStreamWriter2) null);
      ToXmlGenerator toXmlGenerator0 = new ToXmlGenerator((IOContext) null, 2, 1, objectMapper0, streamWriter2Delegate0);
      ToXmlGenerator.Feature toXmlGenerator_Feature0 = ToXmlGenerator.Feature.WRITE_XML_DECLARATION;
      ToXmlGenerator toXmlGenerator1 = toXmlGenerator0.disable(toXmlGenerator_Feature0);
      QName qName0 = new QName("JSON");
      xmlSerializerProvider0._initWithRootName(toXmlGenerator1, qName0);
      xmlSerializerProvider0._initWithRootName(toXmlGenerator0, qName0);
      assertTrue(toXmlGenerator0.canWriteFormattedNumbers());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      StreamWriter2Delegate streamWriter2Delegate0 = new StreamWriter2Delegate((XMLStreamWriter2) null);
      ToXmlGenerator toXmlGenerator0 = new ToXmlGenerator((IOContext) null, 2, 1, objectMapper0, streamWriter2Delegate0);
      toXmlGenerator0.writeStartArray(1);
      ToXmlGenerator.Feature toXmlGenerator_Feature0 = ToXmlGenerator.Feature.WRITE_XML_DECLARATION;
      toXmlGenerator0.disable(toXmlGenerator_Feature0);
      QName qName0 = new QName("JSON");
      xmlSerializerProvider0._initWithRootName(toXmlGenerator0, qName0);
      xmlSerializerProvider0._initWithRootName(toXmlGenerator0, qName0);
      assertEquals(56320, GeneratorBase.SURR2_FIRST);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      StreamWriter2Delegate streamWriter2Delegate0 = new StreamWriter2Delegate((XMLStreamWriter2) null);
      ToXmlGenerator toXmlGenerator0 = new ToXmlGenerator((IOContext) null, 2, 1, objectMapper0, streamWriter2Delegate0);
      ToXmlGenerator.Feature toXmlGenerator_Feature0 = ToXmlGenerator.Feature.WRITE_XML_DECLARATION;
      toXmlGenerator0.disable(toXmlGenerator_Feature0);
      QName qName0 = new QName("JSON", "JSON", "JSON");
      // Undeclared exception!
      try { 
        xmlSerializerProvider0._initWithRootName(toXmlGenerator0, qName0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.codehaus.stax2.util.StreamWriterDelegate", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      // Undeclared exception!
      try { 
        xmlSerializerProvider0.serializeValue((JsonGenerator) null, (Object) javaType0, javaType0, (JsonSerializer<Object>) beanSerializer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.ser.XmlSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      SQLTransientException sQLTransientException0 = new SQLTransientException((Throwable) null);
      IOException iOException0 = xmlSerializerProvider0._wrapAsIOE((JsonGenerator) null, sQLTransientException0);
      assertNotNull(iOException0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      TokenBuffer tokenBuffer0 = new TokenBuffer(jsonParser0);
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      JavaType javaType0 = TypeFactory.unknownType();
      try { 
        xmlSerializerProvider0.serializeValue((JsonGenerator) tokenBuffer0, (Object) tokenBuffer0, javaType0, xmlSerializerProvider0.DEFAULT_NULL_KEY_SERIALIZER);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Null key for a Map not allowed in JSON (use a converting NullKeySerializer?)
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      XmlRootNameLookup xmlRootNameLookup0 = new XmlRootNameLookup();
      XmlSerializerProvider xmlSerializerProvider0 = new XmlSerializerProvider(xmlRootNameLookup0);
      SQLWarning sQLWarning0 = new SQLWarning("", "", 0);
      SQLIntegrityConstraintViolationException sQLIntegrityConstraintViolationException0 = new SQLIntegrityConstraintViolationException("", "", sQLWarning0);
      IOException iOException0 = xmlSerializerProvider0._wrapAsIOE((JsonGenerator) null, sQLIntegrityConstraintViolationException0);
      assertNotNull(iOException0);
  }
}