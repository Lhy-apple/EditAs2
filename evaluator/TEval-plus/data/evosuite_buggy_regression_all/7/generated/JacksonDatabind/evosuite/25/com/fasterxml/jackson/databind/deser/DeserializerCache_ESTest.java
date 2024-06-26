/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:56:04 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.AbstractDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerCache;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.node.MissingNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DeserializerCache_ESTest extends DeserializerCache_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      int int0 = deserializerCache0.cachedDeserializersCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      Object object0 = deserializerCache0.writeReplace();
      assertSame(object0, deserializerCache0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      deserializerCache0.flushCachedDeserializers();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      try { 
        deserializerCache0._handleUnknownKeyDeserializer((JavaType) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not find a (Map) Key deserializer for type null
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      JavaType javaType0 = mapLikeType0.containedTypeOrUnknown(1);
      objectMapper0.canDeserialize(javaType0, (AtomicReference<Throwable>) null);
      ObjectReader objectReader0 = objectMapper0.readerFor(javaType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      Class<DoubleNode> class0 = DoubleNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class0, simpleType0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) collectionLikeType0);
      ObjectReader objectReader1 = objectReader0.forType((JavaType) simpleType0);
      assertFalse(objectReader1.equals((Object)objectReader0));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      JavaType javaType0 = mapLikeType0.containedTypeOrUnknown((-21));
      objectMapper0.canDeserialize(javaType0, (AtomicReference<Throwable>) null);
      boolean boolean0 = objectMapper0.canDeserialize(javaType0, (AtomicReference<Throwable>) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MissingNode> class0 = MissingNode.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      boolean boolean0 = objectMapper0.canDeserialize((JavaType) mapLikeType0, (AtomicReference<Throwable>) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        deserializerCache0.hasValueDeserializerFor(defaultDeserializationContext_Impl0, beanDeserializerFactory0, (JavaType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null JavaType passed
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.NON_FINAL;
      ObjectMapper objectMapper1 = objectMapper0.enableDefaultTypingAsProperty(objectMapper_DefaultTyping0, "JSON");
      boolean boolean0 = objectMapper1.canDeserialize((JavaType) simpleType0, atomicReference0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      // Undeclared exception!
      try { 
        deserializerCache0._createAndCacheValueDeserializer(defaultDeserializationContext_Impl0, beanDeserializerFactory0, simpleType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<JsonInclude.Include> class0 = JsonInclude.Include.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      boolean boolean0 = objectMapper0.canDeserialize((JavaType) mapLikeType0, (AtomicReference<Throwable>) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<AbstractDeserializer> class0 = AbstractDeserializer.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      DeserializerCache deserializerCache0 = new DeserializerCache();
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        deserializerCache0._createDeserializer2(defaultDeserializationContext_Impl0, beanDeserializerFactory0, arrayType0, basicBeanDescription0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedHashMap> class0 = LinkedHashMap.class;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyValueHandler(typeFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      // Undeclared exception!
      try { 
        objectMapper0.canDeserialize((JavaType) mapLikeType1, (AtomicReference<Throwable>) null);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.type.MapLikeType cannot be cast to com.fasterxml.jackson.databind.type.MapType
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      Class<BeanDeserializer> class1 = BeanDeserializer.class;
      Class<String> class2 = String.class;
      MapType mapType0 = MapType.construct(class2, mapLikeType0, mapLikeType0);
      CollectionType collectionType0 = CollectionType.construct(class1, mapType0);
      Object object0 = new Object();
      CollectionLikeType collectionLikeType0 = collectionType0.withContentValueHandler(object0);
      // Undeclared exception!
      try { 
        objectMapper0.readerFor((JavaType) collectionLikeType0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Object cannot be cast to com.fasterxml.jackson.databind.JsonDeserializer
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.TYPE;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) mapLikeType0);
      assertNotNull(objectReader0);
  }
}
