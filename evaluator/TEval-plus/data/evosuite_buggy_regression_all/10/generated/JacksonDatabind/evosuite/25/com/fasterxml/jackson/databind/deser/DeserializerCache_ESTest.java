/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:28:27 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerCache;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.util.EnumSet;
import java.util.NoSuchElementException;
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
      assertSame(deserializerCache0, object0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      deserializerCache0.flushCachedDeserializers();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      try { 
        deserializerCache0._handleUnknownKeyDeserializer(collectionType0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not find a (Map) Key deserializer for type [collection type; class java.util.EnumSet, contains [simple type, class java.lang.Object]]
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<BeanDeserializer> class0 = BeanDeserializer.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      PropertyAccessor propertyAccessor0 = PropertyAccessor.ALL;
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.NON_PRIVATE;
      ObjectMapper objectMapper1 = objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      ObjectReader objectReader0 = objectMapper1.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<NullNode> class0 = NullNode.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      // Undeclared exception!
      try { 
        deserializerCache0.hasValueDeserializerFor(defaultDeserializationContext_Impl0, beanDeserializerFactory0, simpleType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
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
  public void test07()  throws Throwable  {
      Class<BeanDeserializer> class0 = BeanDeserializer.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.NON_FINAL;
      objectMapper0.enableDefaultTypingAsProperty(objectMapper_DefaultTyping0, "");
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[2];
      MapperFeature mapperFeature0 = MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES;
      mapperFeatureArray0[0] = mapperFeature0;
      mapperFeatureArray0[1] = mapperFeatureArray0[0];
      ObjectMapper objectMapper1 = objectMapper0.enable(mapperFeatureArray0);
      // Undeclared exception!
      try { 
        objectMapper1.readerFor(class0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // No entry 'knownPropertyNames' found, can't replace
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.BeanPropertyMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      Class<Module> class0 = Module.class;
      Class<BuilderBasedDeserializer> class1 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      MapType mapType0 = MapType.construct(class1, simpleType0, simpleType0);
      // Undeclared exception!
      try { 
        deserializerCache0._createDeserializer2(defaultDeserializationContext_Impl0, beanDeserializerFactory0, mapType0, basicBeanDescription0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      Class<BeanDeserializer> class1 = BeanDeserializer.class;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      MapLikeType mapLikeType0 = MapLikeType.construct(class1, simpleType0, simpleType0);
      CollectionType collectionType0 = CollectionType.construct(class0, mapLikeType0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) collectionType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<NullNode> class0 = NullNode.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      Class<BeanDeserializer> class1 = BeanDeserializer.class;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      MapLikeType mapLikeType0 = MapLikeType.construct(class1, simpleType0, simpleType0);
      CollectionType collectionType0 = CollectionType.construct(class0, mapLikeType0);
      LongNode longNode0 = LongNode.valueOf(0L);
      CollectionType collectionType1 = collectionType0.withContentValueHandler(longNode0);
      // Undeclared exception!
      try { 
        objectMapper0.readerFor((JavaType) collectionType1);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.node.LongNode cannot be cast to com.fasterxml.jackson.databind.JsonDeserializer
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DeserializerCache deserializerCache0 = new DeserializerCache();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      try { 
        deserializerCache0._handleUnknownValueDeserializer(collectionType0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not find a Value deserializer for abstract type [collection type; class java.util.EnumSet, contains [simple type, class java.lang.Object]]
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }
}