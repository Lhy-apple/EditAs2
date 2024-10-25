/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:56:39 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.cfg.SerializerFactoryConfig;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BeanPropertyDefinition;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.module.SimpleSerializers;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.SerializerFactory;
import com.fasterxml.jackson.databind.ser.Serializers;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.annotation.Annotation;
import java.time.temporal.ChronoField;
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
  public void test00()  throws Throwable  {
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      // Undeclared exception!
      try { 
        beanSerializerFactory0.instance.constructFilteredBeanWriter(beanPropertyWriter0, (Class<?>[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.impl.FilteredBeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withConfig((SerializerFactoryConfig) null);
      assertNotSame(serializerFactory0, beanSerializerFactory0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withConfig(serializerFactoryConfig0);
      assertSame(serializerFactory0, beanSerializerFactory0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      JavaType javaType0 = TypeFactory.unknownType();
      Class<SimpleType> class0 = SimpleType.class;
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
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
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, mapType0, basicBeanDescription0, true);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
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
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class0, mapType0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, collectionLikeType0, basicBeanDescription0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
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
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class0, mapType0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, collectionLikeType0, basicBeanDescription0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
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
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      SimpleSerializers simpleSerializers0 = new SimpleSerializers();
      SerializerFactoryConfig serializerFactoryConfig1 = serializerFactoryConfig0.withAdditionalSerializers(simpleSerializers0);
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig1);
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
  public void test07()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Annotation> class0 = Annotation.class;
      ObjectReader objectReader0 = objectMapper0.reader((Class<?>) class0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<Serializers.Base> class1 = Serializers.Base.class;
      Class<ObjectIdGenerators.IntSequenceGenerator> class2 = ObjectIdGenerators.IntSequenceGenerator.class;
      Class<PropertyName> class3 = PropertyName.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class1, class2, class3);
      // Undeclared exception!
      try { 
        objectMapper0.convertValue((Object) objectMapper0, (JavaType) mapLikeType0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // Could not initialize class com.fasterxml.jackson.databind.JsonMappingException
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.BeanSerializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.PROPERTY;
      byte[] byteArray0 = objectMapper0.writeValueAsBytes(jsonTypeInfo_As0);
      assertEquals(10, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      ObjectMapper objectMapper0 = new ObjectMapper();
      Iterable<Serializers> iterable0 = serializerFactoryConfig0.keySerializers();
      byte[] byteArray0 = objectMapper0.writeValueAsBytes(iterable0);
      assertEquals(2, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ChronoField> class0 = ChronoField.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      CollectionType collectionType0 = CollectionType.construct(class0, simpleType0);
      MapType mapType0 = MapType.construct(class0, collectionType0, simpleType0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0.findBeanSerializer(defaultSerializerProvider_Impl0, mapType0, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Integer> class0 = Integer.class;
      ObjectReader objectReader0 = objectMapper0.readerWithView((Class<?>) class0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<Object> jsonSerializer0 = beanSerializerFactory0.findBeanSerializer(defaultSerializerProvider_Impl0, arrayType0, (BeanDescription) null);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Object object0 = new Object();
      // Undeclared exception!
      try { 
        objectMapper0.writeValueAsBytes(object0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // Could not initialize class com.fasterxml.jackson.databind.JsonMappingException
         //
         verifyException("com.fasterxml.jackson.databind.ser.impl.UnknownSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      LinkedList<BeanPropertyDefinition> linkedList0 = new LinkedList<BeanPropertyDefinition>();
      beanSerializerFactory0.instance.removeSetterlessGetters((SerializationConfig) null, (BeanDescription) null, linkedList0);
      assertEquals(0, linkedList0.size());
  }
}
