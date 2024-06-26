/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:40:03 GMT 2023
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
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.cfg.SerializerFactoryConfig;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BeanPropertyDefinition;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.SerializerFactory;
import com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassKey;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.reflect.Array;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanSerializerFactory_ESTest extends BeanSerializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withConfig(serializerFactoryConfig0);
      assertNotSame(serializerFactory0, beanSerializerFactory0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      BeanSerializerBuilder beanSerializerBuilder0 = beanSerializerFactory0.constructBeanSerializerBuilder((BeanDescription) null);
      assertFalse(beanSerializerBuilder0.hasProperties());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      // Undeclared exception!
      try { 
        beanSerializerFactory0.constructPropertyBuilder((SerializationConfig) null, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.PropertyBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<ClassKey>[] classArray0 = (Class<ClassKey>[]) Array.newInstance(Class.class, 6);
      BeanPropertyWriter beanPropertyWriter1 = beanSerializerFactory0.constructFilteredBeanWriter(beanPropertyWriter0, classArray0);
      assertFalse(beanPropertyWriter1.isVirtual());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<SimpleModule> class0 = SimpleModule.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMember) null).when(pOJOPropertiesCollector0).getAnyGetter();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
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
  public void test5()  throws Throwable  {
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withConfig(serializerFactoryConfig0);
      assertSame(serializerFactory0, beanSerializerFactory0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<String> class0 = String.class;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      ArrayType arrayType0 = ArrayType.construct(javaType0, typeFactory0, beanSerializerFactory0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, arrayType0, javaType0);
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
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, mapLikeType0, basicBeanDescription0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = TypeFactory.unknownType();
      ArrayType arrayType0 = ArrayType.construct(javaType0, typeFactory0, beanSerializerFactory0);
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
      JsonSerializer<Object> jsonSerializer0 = beanSerializerFactory0.findBeanSerializer(defaultSerializerProvider_Impl0, arrayType0, basicBeanDescription0);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
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
      LinkedList<BeanPropertyWriter> linkedList0 = new LinkedList<BeanPropertyWriter>();
      ObjectIdWriter objectIdWriter0 = beanSerializerFactory0.constructObjectIdHandler((SerializerProvider) null, basicBeanDescription0, linkedList0);
      assertNull(objectIdWriter0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, annotationIntrospector0, false);
      LinkedList<BeanPropertyDefinition> linkedList0 = new LinkedList<BeanPropertyDefinition>();
      linkedList0.add((BeanPropertyDefinition) pOJOPropertyBuilder0);
      beanSerializerFactory0.removeSetterlessGetters((SerializationConfig) null, (BeanDescription) null, linkedList0);
      assertEquals(0, linkedList0.size());
  }
}
