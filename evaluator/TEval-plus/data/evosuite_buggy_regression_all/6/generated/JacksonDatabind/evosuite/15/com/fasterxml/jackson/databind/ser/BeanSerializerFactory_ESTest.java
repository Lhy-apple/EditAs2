/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:21:02 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.cfg.SerializerFactoryConfig;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BeanPropertyDefinition;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.SerializerFactory;
import com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter;
import com.fasterxml.jackson.databind.type.ClassKey;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.lang.reflect.Array;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanSerializerFactory_ESTest extends BeanSerializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withConfig(serializerFactoryConfig0);
      assertSame(serializerFactory0, beanSerializerFactory0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      BeanSerializerBuilder beanSerializerBuilder0 = beanSerializerFactory0.constructBeanSerializerBuilder((BeanDescription) null);
      assertFalse(beanSerializerBuilder0.hasProperties());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<ClassKey> class0 = ClassKey.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationIntrospector annotationIntrospector1 = AnnotationIntrospector.pair(annotationIntrospector0, annotationIntrospector0);
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector1, (ClassIntrospector.MixInResolver) null);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, (JavaType) null, annotatedClass0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0.instance.constructPropertyBuilder((SerializationConfig) null, basicBeanDescription0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.PropertyBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<ObjectReader>[] classArray0 = (Class<ObjectReader>[]) Array.newInstance(Class.class, 2);
      // Undeclared exception!
      try { 
        beanSerializerFactory0.constructFilteredBeanWriter((BeanPropertyWriter) null, classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<NamedType> class0 = NamedType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getAnySetterMethod();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((Set) null).when(pOJOPropertiesCollector0).getIgnoredPropertyNames();
      doReturn((Map) null).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.BeanDescription", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withConfig(serializerFactoryConfig0);
      assertNotSame(serializerFactory0, beanSerializerFactory0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<NullNode> class0 = NullNode.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, (BeanDescription) null, true);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getAnySetterMethod();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((Set) null).when(pOJOPropertiesCollector0).getIgnoredPropertyNames();
      doReturn((Map) null).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, mapType0, basicBeanDescription0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      CollectionType collectionType0 = CollectionType.construct(class0, mapType0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getAnySetterMethod();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((Set) null).when(pOJOPropertiesCollector0).getIgnoredPropertyNames();
      doReturn((Map) null).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, collectionType0, basicBeanDescription0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getAnySetterMethod();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((Set) null).when(pOJOPropertiesCollector0).getIgnoredPropertyNames();
      doReturn((Map) null).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, true);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getAnySetterMethod();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((Set) null).when(pOJOPropertiesCollector0).getIgnoredPropertyNames();
      doReturn((Map) null).when(pOJOPropertiesCollector0).getInjectables();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      Vector<BeanPropertyWriter> vector0 = new Vector<BeanPropertyWriter>();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectIdWriter objectIdWriter0 = beanSerializerFactory0.constructObjectIdHandler(defaultSerializerProvider_Impl0, basicBeanDescription0, vector0);
      assertNull(objectIdWriter0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<Boolean> class0 = Boolean.TYPE;
      boolean boolean0 = beanSerializerFactory0.isPotentialBeanType(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      PropertyName propertyName0 = new PropertyName("");
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, annotationIntrospector0, true);
      Stack<BeanPropertyDefinition> stack0 = new Stack<BeanPropertyDefinition>();
      stack0.add((BeanPropertyDefinition) pOJOPropertyBuilder0);
      beanSerializerFactory0.removeSetterlessGetters((SerializationConfig) null, (BeanDescription) null, stack0);
      assertEquals(0, stack0.size());
      assertTrue(stack0.empty());
  }
}
