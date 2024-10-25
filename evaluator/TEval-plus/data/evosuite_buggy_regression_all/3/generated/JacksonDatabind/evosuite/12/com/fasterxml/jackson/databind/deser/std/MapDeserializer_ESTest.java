/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:37:03 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.KeyDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.FromStringDeserializer;
import com.fasterxml.jackson.databind.deser.std.JsonLocationInstantiator;
import com.fasterxml.jackson.databind.deser.std.MapDeserializer;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializer;
import com.fasterxml.jackson.databind.deser.std.StdValueInstantiator;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.util.Annotations;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.time.chrono.ChronoLocalDate;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockIOException;
import org.evosuite.runtime.mock.java.lang.MockError;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MapDeserializer_ESTest extends MapDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      // Undeclared exception!
      try { 
        mapDeserializer0.deserializeWithType((JsonParser) null, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MapDeserializer mapDeserializer0 = null;
      try {
        mapDeserializer0 = new MapDeserializer((MapDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      Class<Object> class1 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class1);
      JsonDeserializer<JsonAutoDetect.Visibility> jsonDeserializer0 = (JsonDeserializer<JsonAutoDetect.Visibility>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class1, (-2039));
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, stdValueInstantiator0, stdKeyDeserializer_DelegatingKD0, coreXMLDeserializers_Std0, (TypeDeserializer) null);
      JavaType javaType0 = mapDeserializer0.getValueType();
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.reader(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      JsonDeserializer<Object> jsonDeserializer0 = mapDeserializer0.getContentDeserializer();
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        mapDeserializer0.deserialize((JsonParser) null, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      Class<Integer> class1 = Integer.class;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class1, 215);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(mapType0, (TypeIdResolver) null, "=u#qxM-R~<0", true, class1);
      MapDeserializer mapDeserializer0 = new MapDeserializer(mapType0, stdValueInstantiator0, stdKeyDeserializer_DelegatingKD0, fromStringDeserializer_Std0, asPropertyTypeDeserializer0);
      mapDeserializer0.getContentType();
      assertFalse(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, (JsonDeserializer<?>) null, linkedHashSet0);
      assertFalse(mapDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(1116);
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, mapDeserializer0, linkedHashSet0);
      assertFalse(mapDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Class<JsonAutoDetect.Visibility> class0 = JsonAutoDetect.Visibility.class;
      StdKeyDeserializer stdKeyDeserializer0 = new StdKeyDeserializer((-567), class0);
      Class<Object> class1 = Object.class;
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(simpleType0, (TypeIdResolver) null, "", true, class1);
      AsExternalTypeDeserializer asExternalTypeDeserializer1 = new AsExternalTypeDeserializer(asExternalTypeDeserializer0, (BeanProperty) null);
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, stdKeyDeserializer0, (JsonDeserializer<Object>) null, asExternalTypeDeserializer1);
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, mapDeserializer0);
      HashSet<String> hashSet0 = new HashSet<String>();
      MapDeserializer mapDeserializer1 = new MapDeserializer(mapDeserializer0, stdKeyDeserializer_DelegatingKD0, (JsonDeserializer<Object>) null, asExternalTypeDeserializer0, hashSet0);
      MapDeserializer mapDeserializer2 = mapDeserializer1.withResolved(stdKeyDeserializer_DelegatingKD0, asExternalTypeDeserializer1, (JsonDeserializer<?>) null, hashSet0);
      assertNotSame(mapDeserializer2, mapDeserializer1);
      assertFalse(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      MapDeserializer mapDeserializer1 = mapDeserializer0.withResolved((KeyDeserializer) null, (TypeDeserializer) null, (JsonDeserializer<?>) null, (HashSet<String>) null);
      assertSame(mapDeserializer1, mapDeserializer0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<String> class0 = String.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      JsonDeserializer<ChronoLocalDate> jsonDeserializer0 = (JsonDeserializer<ChronoLocalDate>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, jsonDeserializer0);
      Class<ByteArrayInputStream> class1 = ByteArrayInputStream.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class1);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      MapDeserializer mapDeserializer0 = new MapDeserializer(mapType0, stdValueInstantiator0, stdKeyDeserializer_DelegatingKD0, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      assertTrue(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      String[] stringArray0 = new String[1];
      mapDeserializer0.setIgnorableProperties(stringArray0);
      assertFalse(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      String[] stringArray0 = new String[0];
      mapDeserializer0.setIgnorableProperties(stringArray0);
      assertTrue(mapDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      // Undeclared exception!
      try { 
        mapDeserializer0.resolve((DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.MapDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      Class<CreatorProperty> class0 = CreatorProperty.class;
      StdKeyDeserializer.DelegatingKD stdKeyDeserializer_DelegatingKD0 = new StdKeyDeserializer.DelegatingKD(class0, (JsonDeserializer<?>) null);
      Class<Integer> class1 = Integer.TYPE;
      FromStringDeserializer.Std fromStringDeserializer_Std0 = new FromStringDeserializer.Std(class1, 29);
      Class<Error> class2 = Error.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "=u#qxM-R~<0", false, class2);
      Class<ChronoLocalDate> class3 = ChronoLocalDate.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class3);
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, stdValueInstantiator0, stdKeyDeserializer_DelegatingKD0, fromStringDeserializer_Std0, asPropertyTypeDeserializer0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, asPropertyTypeDeserializer0, (Annotations) null, (AnnotatedParameter) null, (-544), (Object) null, (PropertyMetadata) null);
      // Undeclared exception!
      try { 
        mapDeserializer0.createContextual(deserializationContext0, creatorProperty0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.enableDefaultTyping();
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.reader(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(0);
      MapDeserializer mapDeserializer1 = new MapDeserializer(mapDeserializer0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, linkedHashSet0);
      boolean boolean0 = mapDeserializer1.isCachable();
      assertTrue(mapDeserializer0.isCachable());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer(simpleType0, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      // Undeclared exception!
      try { 
        mapDeserializer0.deserialize(jsonParser0, deserializationContext0, (Map<Object, Object>) hashMap0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Class<IOException> class0 = IOException.class;
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, (-2804));
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, coreXMLDeserializers_Std0, (TypeDeserializer) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ReaderBasedJsonParser readerBasedJsonParser0 = (ReaderBasedJsonParser)jsonFactory0.createParser("JSON");
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectMapper0.getDeserializationContext();
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      mapDeserializer0._readAndBind(readerBasedJsonParser0, defaultDeserializationContext_Impl0, hashMap0);
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Class<IOException> class0 = IOException.class;
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, (-2804));
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, coreXMLDeserializers_Std0, (TypeDeserializer) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ReaderBasedJsonParser readerBasedJsonParser0 = (ReaderBasedJsonParser)jsonFactory0.createParser("JSON");
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectMapper0.getDeserializationContext();
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      mapDeserializer0._readAndBindStringMap(readerBasedJsonParser0, defaultDeserializationContext_Impl0, hashMap0);
      assertEquals(0L, readerBasedJsonParser0.getTokenCharacterOffset());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      InvocationTargetException invocationTargetException0 = new InvocationTargetException((Throwable) null, "JSON");
      try { 
        mapDeserializer0.wrapAndThrow(invocationTargetException0, invocationTargetException0, "JSON");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // JSON (through reference chain: java.lang.reflect.InvocationTargetException[\"JSON\"])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException();
      InvocationTargetException invocationTargetException0 = new InvocationTargetException(sQLInvalidAuthorizationSpecException0);
      try { 
        mapDeserializer0.wrapAndThrow(invocationTargetException0, invocationTargetException0, "}@");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // (was java.sql.SQLInvalidAuthorizationSpecException) (through reference chain: java.lang.reflect.InvocationTargetException[\"}@\"])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException();
      MockError mockError0 = new MockError(sQLInvalidAuthorizationSpecException0);
      // Undeclared exception!
      try { 
        mapDeserializer0.wrapAndThrow(mockError0, (Object) null, "&u` BKhT)<!\"");
        fail("Expecting exception: Error");
      
      } catch(Error e) {
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("!+R@#D;KFXH", ": value instantiator (");
      InvocationTargetException invocationTargetException0 = new InvocationTargetException(sQLInvalidAuthorizationSpecException0, ": value instantiator (");
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(byteArrayInputStream0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) invocationTargetException0, jsonMappingException_Reference0);
      // Undeclared exception!
      try { 
        mapDeserializer0.wrapAndThrow(jsonMappingException0, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Can not pass null fieldName
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      MapDeserializer mapDeserializer0 = new MapDeserializer((JavaType) null, jsonLocationInstantiator0, (KeyDeserializer) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null);
      MockIOException mockIOException0 = new MockIOException("hu=D~VWDs.;#U");
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException();
      try { 
        mapDeserializer0.wrapAndThrow(mockIOException0, sQLInvalidAuthorizationSpecException0, "hu=D~VWDs.;#U");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
      }
  }
}
