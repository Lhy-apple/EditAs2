/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:27:33 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.cfg.SerializerFactoryConfig;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.jsontype.TypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BigIntegerNode;
import com.fasterxml.jackson.databind.node.BooleanNode;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.node.IntNode;
import com.fasterxml.jackson.databind.node.ShortNode;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.BeanSerializerModifier;
import com.fasterxml.jackson.databind.ser.ContainerSerializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.SerializerFactory;
import com.fasterxml.jackson.databind.ser.Serializers;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.net.InetAddress;
import java.nio.CharBuffer;
import java.util.Map;
import java.util.SimpleTimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BasicSerializerFactory_ESTest extends BasicSerializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Map> class0 = Map.class;
      Class<BooleanNode> class1 = BooleanNode.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class1);
      // Undeclared exception!
      try { 
        beanSerializerFactory0.buildIteratorSerializer((SerializationConfig) null, mapType0, (BeanDescription) null, true, mapType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      BeanSerializerModifier beanSerializerModifier0 = mock(BeanSerializerModifier.class, new ViolatedAssumptionAnswer());
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withSerializerModifier(beanSerializerModifier0);
      assertFalse(serializerFactory0.equals((Object)beanSerializerFactory0));
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      // Undeclared exception!
      try { 
        beanSerializerFactory0.findFilterId((SerializationConfig) null, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      // Undeclared exception!
      try { 
        beanSerializerFactory0.buildMapEntrySerializer((SerializationConfig) null, (JavaType) null, (BeanDescription) null, false, (JavaType) null, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Serializers.Base serializers_Base0 = new Serializers.Base();
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withAdditionalSerializers(serializers_Base0);
      assertNotSame(beanSerializerFactory0, serializerFactory0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<BooleanNode> class0 = BooleanNode.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayNode> class1 = ArrayNode.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class1);
      MapType mapType0 = MapType.construct(class0, collectionLikeType0, collectionLikeType0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionLikeType0, typeFactory0);
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      AsWrapperTypeSerializer asWrapperTypeSerializer0 = new AsWrapperTypeSerializer(classNameIdResolver0, beanPropertyWriter0);
      ContainerSerializer<?> containerSerializer0 = beanSerializerFactory0.buildIndexedListSerializer(mapType0, true, asWrapperTypeSerializer0, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      assertFalse(containerSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<SimpleTimeZone> class0 = SimpleTimeZone.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.findSerializerByPrimaryType(defaultSerializerProvider_Impl0, simpleType0, (BeanDescription) null, true);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<DecimalNode> class0 = DecimalNode.class;
      boolean boolean0 = beanSerializerFactory0.isIndexedList(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      SerializerFactoryConfig serializerFactoryConfig0 = beanSerializerFactory0.getFactoryConfig();
      assertFalse(serializerFactoryConfig0.hasSerializerModifiers());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Serializers.Base serializers_Base0 = new Serializers.Base();
      SerializerFactory serializerFactory0 = beanSerializerFactory0.withAdditionalKeySerializers(serializers_Base0);
      assertFalse(serializerFactory0.equals((Object)beanSerializerFactory0));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      JavaType javaType0 = TypeFactory.unknownType();
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.buildEnumSetSerializer(javaType0);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory((SerializerFactoryConfig) null);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<IntNode> class0 = IntNode.class;
      Class<SimpleTimeZone> class1 = SimpleTimeZone.class;
      SimpleType simpleType0 = SimpleType.construct(class1);
      CollectionType collectionType0 = CollectionType.construct(class0, simpleType0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.findSerializerByAnnotations(defaultSerializerProvider_Impl0, collectionType0, (BeanDescription) null);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((AnnotatedMethod) null).when(pOJOPropertiesCollector0).getJsonValueMethod();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.findSerializerByAnnotations((SerializerProvider) null, simpleType0, basicBeanDescription0);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<CharBuffer> class0 = CharBuffer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<InetAddress> class1 = InetAddress.class;
      CollectionType collectionType0 = CollectionType.construct(class1, simpleType0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.findSerializerByPrimaryType(defaultSerializerProvider_Impl0, collectionType0, (BeanDescription) null, true);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<StringBuffer> class0 = StringBuffer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionType collectionType0 = CollectionType.construct(class0, simpleType0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.findSerializerByPrimaryType(defaultSerializerProvider_Impl0, collectionType0, (BeanDescription) null, false);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<CharBuffer> class0 = CharBuffer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<InetAddress> class1 = InetAddress.class;
      CollectionType collectionType0 = CollectionType.construct(class1, simpleType0);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.findSerializerByAddonType((SerializationConfig) null, collectionType0, (BeanDescription) null, false);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<ShortNode> class0 = ShortNode.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0.findSerializerByAddonType((SerializationConfig) null, simpleType0, (BeanDescription) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<StringBuffer> class0 = StringBuffer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionType collectionType0 = CollectionType.construct(class0, simpleType0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        beanSerializerFactory0.buildContainerSerializer(defaultSerializerProvider_Impl0, collectionType0, (BeanDescription) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      ArrayType arrayType0 = ArrayType.construct(simpleType0, (Object) null, defaultSerializerProvider_Impl0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0.buildContainerSerializer(defaultSerializerProvider_Impl0, arrayType0, (BeanDescription) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      ArrayType arrayType0 = ArrayType.construct(simpleType0, (Object) null, defaultSerializerProvider_Impl0);
      ArrayType arrayType1 = arrayType0.withStaticTyping();
      // Undeclared exception!
      try { 
        beanSerializerFactory0.buildContainerSerializer(defaultSerializerProvider_Impl0, arrayType1, (BeanDescription) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapType> class0 = MapType.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) simpleType0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0.buildArraySerializer((SerializationConfig) null, arrayType0, basicBeanDescription0, true, (TypeSerializer) null, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<MockGregorianCalendar> class0 = MockGregorianCalendar.class;
      Class<BigIntegerNode> class1 = BigIntegerNode.class;
      Class<?> class2 = beanSerializerFactory0._verifyAsClass(class0, "", class1);
      assertEquals("class org.evosuite.runtime.mock.java.util.MockGregorianCalendar", class2.toString());
      assertNotNull(class2);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<BeanSerializer> class0 = BeanSerializer.class;
      Class<?> class1 = beanSerializerFactory0._verifyAsClass((Object) null, "Ki", class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<MockGregorianCalendar> class0 = MockGregorianCalendar.class;
      // Undeclared exception!
      try { 
        beanSerializerFactory0._verifyAsClass("", "", class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector.() returned value of type java.lang.String: expected type JsonSerializer or Class<JsonSerializer> instead
         //
         verifyException("com.fasterxml.jackson.databind.ser.BasicSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<MockGregorianCalendar> class0 = MockGregorianCalendar.class;
      Class<?> class1 = beanSerializerFactory0._verifyAsClass(class0, "", class0);
      assertNull(class1);
  }
}