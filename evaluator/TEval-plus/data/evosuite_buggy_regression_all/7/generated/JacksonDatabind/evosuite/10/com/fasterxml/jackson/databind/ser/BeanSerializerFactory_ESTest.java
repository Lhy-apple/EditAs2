/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:53:03 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
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
import com.fasterxml.jackson.databind.module.SimpleSerializers;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.BeanSerializerModifier;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.SerializerFactory;
import com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.reflect.Array;
import java.time.chrono.MinguoEra;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
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
  public void test03()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<ObjectIdGenerators.IntSequenceGenerator>[] classArray0 = (Class<ObjectIdGenerators.IntSequenceGenerator>[]) Array.newInstance(Class.class, 2);
      BeanPropertyWriter beanPropertyWriter1 = beanSerializerFactory0.instance.constructFilteredBeanWriter(beanPropertyWriter0, classArray0);
      assertFalse(beanPropertyWriter1.hasNullSerializer());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withConfig(serializerFactoryConfig0);
      assertNotSame(serializerFactory0, beanSerializerFactory0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<ArrayType> class0 = ArrayType.class;
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
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2((SerializerProvider) null, simpleType0, basicBeanDescription0, false);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
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
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<String> class1 = String.class;
      MapType mapType0 = MapType.construct(class1, simpleType0, simpleType0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, mapType0, basicBeanDescription0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
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
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<String> class1 = String.class;
      MapType mapType0 = MapType.construct(class1, simpleType0, simpleType0);
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
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SimpleSerializers simpleSerializers0 = new SimpleSerializers();
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      SerializerFactoryConfig serializerFactoryConfig1 = serializerFactoryConfig0.withAdditionalSerializers(simpleSerializers0);
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig1);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.BeanDescription", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, false);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerModifier beanSerializerModifier0 = mock(BeanSerializerModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JsonSerializer) null).when(beanSerializerModifier0).modifySerializer(any(com.fasterxml.jackson.databind.SerializationConfig.class) , any(com.fasterxml.jackson.databind.BeanDescription.class) , any(com.fasterxml.jackson.databind.JsonSerializer.class));
      SerializerFactoryConfig serializerFactoryConfig1 = serializerFactoryConfig0.withSerializerModifier(beanSerializerModifier0);
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig1);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, false);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
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
      Class<MinguoEra> class0 = MinguoEra.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        beanSerializerFactory0.findBeanSerializer(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.BeanDescription", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MinguoEra> class0 = MinguoEra.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMember) null).when(pOJOPropertiesCollector0).getAnyGetter();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      JsonSerializer<Object> jsonSerializer0 = beanSerializerFactory0.findBeanSerializer(defaultSerializerProvider_Impl0, arrayType0, basicBeanDescription0);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedMember) null).when(pOJOPropertiesCollector0).getAnyGetter();
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((List) null).when(pOJOPropertiesCollector0).getProperties();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      ArrayList<BeanPropertyWriter> arrayList0 = new ArrayList<BeanPropertyWriter>();
      ObjectIdWriter objectIdWriter0 = beanSerializerFactory0.constructObjectIdHandler((SerializerProvider) null, basicBeanDescription0, arrayList0);
      assertNull(objectIdWriter0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Stack<BeanPropertyDefinition> stack0 = new Stack<BeanPropertyDefinition>();
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder((PropertyName) null, annotationIntrospector0, true);
      stack0.add((BeanPropertyDefinition) pOJOPropertyBuilder0);
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      beanSerializerFactory0.removeSetterlessGetters((SerializationConfig) null, (BeanDescription) null, stack0);
      assertTrue(stack0.empty());
      assertEquals("[]", stack0.toString());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Stack<BeanPropertyDefinition> stack0 = new Stack<BeanPropertyDefinition>();
      PropertyName propertyName0 = new PropertyName("com.fasterxml.jackson.core.json.ReaderBasedJsonParser", (String) null);
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, false);
      pOJOPropertyBuilder0.addGetter((AnnotatedMethod) null, propertyName0, false, false, false);
      stack0.add((BeanPropertyDefinition) pOJOPropertyBuilder0);
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      beanSerializerFactory0.removeSetterlessGetters((SerializationConfig) null, (BeanDescription) null, stack0);
      assertFalse(stack0.isEmpty());
      assertEquals(1, stack0.size());
  }
}