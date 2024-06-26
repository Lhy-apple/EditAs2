/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:12:13 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.impl.UnknownSerializer;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.io.OutputStream;
import java.sql.BatchUpdateException;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.sql.SQLNonTransientConnectionException;
import java.sql.SQLTimeoutException;
import java.sql.SQLTransientException;
import java.sql.SQLWarning;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.text.MockDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SerializerProvider_ESTest extends SerializerProvider_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      Object[] objectArray0 = new Object[6];
      try { 
        serializerProvider0.reportMappingProblem("", objectArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // 
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<String> class0 = String.class;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.getDefaultPropertyFormat(class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.getLocale();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.findTypeSerializer((JavaType) null);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.setAttribute(defaultSerializerProvider_Impl0, objectMapper0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = (DefaultSerializerProvider.Impl)objectMapper0.getSerializerProviderInstance();
      SQLTransientException sQLTransientException0 = new SQLTransientException();
      Object[] objectArray0 = new Object[7];
      JsonMappingException jsonMappingException0 = ((SerializerProvider)defaultSerializerProvider_Impl0).mappingException((Throwable) sQLTransientException0, "<0L>IGu-P'8tR4e6w?0", objectArray0);
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      Class<BeanSerializer> class0 = BeanSerializer.class;
      JsonInclude.Value jsonInclude_Value0 = serializerProvider0.getDefaultPropertyInclusion(class0);
      assertEquals(JsonInclude.Include.USE_DEFAULTS, jsonInclude_Value0.getContentInclusion());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      boolean boolean0 = serializerProvider0.hasSerializationFeatures(2);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<MockDateFormat> class0 = MockDateFormat.class;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.getAttribute(class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      MapperFeature mapperFeature0 = MapperFeature.INFER_CREATOR_FROM_CONSTRUCTOR_PROPERTIES;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.isEnabled(mapperFeature0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedHashSet> class0 = LinkedHashSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      try { 
        defaultSerializerProvider_Impl0.reportBadDefinition((JavaType) collectionType0, "^pwT");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // ^pwT
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<String> class0 = String.class;
      SQLWarning sQLWarning0 = new SQLWarning("-mS[", "5ONi)p?", (-1));
      SQLNonTransientConnectionException sQLNonTransientConnectionException0 = new SQLNonTransientConnectionException("-mS[", "5ONi)p?", sQLWarning0);
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.reportBadDefinition((Class<?>) class0, "-mS[", (Throwable) sQLNonTransientConnectionException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = (DefaultSerializerProvider.Impl)objectMapper0.getSerializerProviderInstance();
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("<60H zX]\"bs2", "<60H zX]\"bs2");
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>(sQLInvalidAuthorizationSpecException0);
      Class<BeanSerializer> class0 = BeanSerializer.class;
      boolean boolean0 = defaultSerializerProvider_Impl0.hasSerializerFor(class0, atomicReference0);
      assertEquals(3, defaultSerializerProvider_Impl0.cachedSerializersCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[3];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0, (-216), (-216));
      Class<Integer> class0 = Integer.class;
      JsonToken jsonToken0 = JsonToken.VALUE_NULL;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.wrongTokenException(jsonParser0, class0, jsonToken0, (String) null);
      try { 
        serializerProvider0.reportBadDefinition((JavaType) null, (String) null, (Throwable) jsonMappingException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // N/A
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper objectMapper1 = new ObjectMapper(objectMapper0);
      assertEquals(0, objectMapper1.mixInCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<?> class0 = defaultSerializerProvider_Impl0.getSerializationView();
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<?> class0 = defaultSerializerProvider_Impl0.getActiveView();
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<Object> jsonSerializer0 = defaultSerializerProvider_Impl0.getDefaultNullValueSerializer();
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.canOverrideAccessModifiers();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonSerializer<Object> jsonSerializer0 = serializerProvider0.getDefaultNullKeySerializer();
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      int[] intArray0 = new int[0];
      BatchUpdateException batchUpdateException0 = new BatchUpdateException("+j", "Zp<aFN8[IA9", 1347, intArray0, (Throwable) null);
      SQLTimeoutException sQLTimeoutException0 = new SQLTimeoutException("[N/A]", batchUpdateException0);
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.reportMappingProblem((Throwable) sQLTimeoutException0, "", (Object[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DatabindContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.getTimeZone();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.getFilterProvider();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      defaultSerializerProvider_Impl0.setNullValueSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      assertEquals(0, defaultSerializerProvider_Impl0.cachedSerializersCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.setNullValueSerializer((JsonSerializer<Object>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot pass null JsonSerializer
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = (DefaultSerializerProvider.Impl)objectMapper0.getSerializerProvider();
      defaultSerializerProvider_Impl0.setNullKeySerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      assertEquals(0, defaultSerializerProvider_Impl0.cachedSerializersCount());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = (DefaultSerializerProvider.Impl)objectMapper0.getSerializerProviderInstance();
      ObjectReader objectReader0 = objectMapper0.reader();
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<Map> class0 = Map.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      defaultSerializerProvider_Impl0.findValueSerializer((JavaType) mapType0);
      assertEquals(1, defaultSerializerProvider_Impl0.cachedSerializersCount());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<UnknownSerializer> class0 = UnknownSerializer.class;
      JsonSerializer<Object> jsonSerializer0 = defaultSerializerProvider_Impl0.getUnknownTypeSerializer(class0);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JsonSerializer<?> jsonSerializer0 = defaultSerializerProvider_Impl0.handleSecondaryContextualization((JsonSerializer<?>) null, beanProperty_Bogus0);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0.CHAR_CONCAT_BUFFER, false);
      MockPrintStream mockPrintStream0 = new MockPrintStream(" of 4-char base64 unit: can only used between units");
      byte[] byteArray0 = new byte[6];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, (-1838), objectMapper0, mockPrintStream0, byteArray0, (byte)115, true);
      serializerProvider0.defaultSerializeDateValue((-1619L), (JsonGenerator) uTF8JsonGenerator0);
      assertArrayEquals(new byte[] {(byte)45, (byte)49, (byte)54, (byte)49, (byte)57, (byte)0}, byteArray0);
      assertEquals(5, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      // Undeclared exception!
      try { 
        serializerProvider0.defaultSerializeDateKey((long) 1, (JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.AUTO_CLOSE_TARGET;
      MockDate mockDate0 = new MockDate();
      IOContext iOContext0 = new IOContext(bufferRecycler0, jsonGenerator_Feature0, false);
      byte[] byteArray0 = new byte[2];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 3, objectMapper0, (OutputStream) null, byteArray0, (byte)3, false);
      try { 
        serializerProvider0.defaultSerializeDateKey((Date) mockDate0, (JsonGenerator) uTF8JsonGenerator0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not write a field name, expecting a value
         //
         verifyException("com.fasterxml.jackson.core.JsonGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, (Object) null, true);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, objectMapper0, (OutputStream) null);
      serializerProvider0.defaultSerializeNull(uTF8JsonGenerator0);
      assertEquals(56320, GeneratorBase.SURR2_FIRST);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.defaultSerializeNull((JsonGenerator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Object[] objectArray0 = new Object[6];
      try { 
        defaultSerializerProvider_Impl0.reportBadTypeDefinition((BeanDescription) null, "X", objectArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Invalid type definition for type N/A: X
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }
}
