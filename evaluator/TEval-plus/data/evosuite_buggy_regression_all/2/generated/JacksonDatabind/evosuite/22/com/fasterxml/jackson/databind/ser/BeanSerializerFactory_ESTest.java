/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:58:14 GMT 2023
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
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BeanPropertyDefinition;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder;
import com.fasterxml.jackson.databind.node.BigIntegerNode;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.BeanSerializerModifier;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.SerializerFactory;
import com.fasterxml.jackson.databind.ser.Serializers;
import com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.reflect.Array;
import java.time.temporal.ChronoUnit;
import java.util.ArrayDeque;
import java.util.LinkedList;
import java.util.List;
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
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, simpleType0, (AnnotatedClass) null);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanSerializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      // Undeclared exception!
      try { 
        beanSerializerFactory0.instance.constructPropertyBuilder((SerializationConfig) null, (BeanDescription) null);
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
      Class<Integer>[] classArray0 = (Class<Integer>[]) Array.newInstance(Class.class, 1);
      BeanPropertyWriter beanPropertyWriter1 = beanSerializerFactory0.constructFilteredBeanWriter(beanPropertyWriter0, classArray0);
      assertFalse(beanPropertyWriter1.hasSerializer());
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
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<BigIntegerNode> class0 = BigIntegerNode.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2((SerializerProvider) null, collectionLikeType0, basicBeanDescription0, true);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayDeque> class0 = ArrayDeque.class;
      Class<ObjectIdGenerators.IntSequenceGenerator> class1 = ObjectIdGenerators.IntSequenceGenerator.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, collectionType0, (AnnotatedClass) null);
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
  public void test07()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, simpleType0, (AnnotatedClass) null);
      ArrayType arrayType0 = ArrayType.construct(simpleType0, class0, class0);
      // Undeclared exception!
      try { 
        beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, arrayType0, basicBeanDescription0, true);
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
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, simpleType0, (AnnotatedClass) null);
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      Serializers.Base serializers_Base0 = new Serializers.Base();
      SerializerFactoryConfig serializerFactoryConfig1 = serializerFactoryConfig0.withAdditionalSerializers(serializers_Base0);
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig1);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, false);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, simpleType0, (AnnotatedClass) null);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, true);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, simpleType0, (AnnotatedClass) null);
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerModifier beanSerializerModifier0 = mock(BeanSerializerModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JsonSerializer) null).when(beanSerializerModifier0).modifySerializer(any(com.fasterxml.jackson.databind.SerializationConfig.class) , any(com.fasterxml.jackson.databind.BeanDescription.class) , any(com.fasterxml.jackson.databind.JsonSerializer.class));
      SerializerFactoryConfig serializerFactoryConfig1 = serializerFactoryConfig0.withSerializerModifier(beanSerializerModifier0);
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig1);
      JsonSerializer<?> jsonSerializer0 = beanSerializerFactory0._createSerializer2(defaultSerializerProvider_Impl0, simpleType0, basicBeanDescription0, true);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forOtherUse((MapperConfig<?>) null, simpleType0, (AnnotatedClass) null);
      LinkedList<BeanPropertyWriter> linkedList0 = new LinkedList<BeanPropertyWriter>();
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      ObjectIdWriter objectIdWriter0 = beanSerializerFactory0.constructObjectIdHandler(defaultSerializerProvider_Impl0, basicBeanDescription0, linkedList0);
      assertNull(objectIdWriter0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Class<ChronoUnit> class0 = ChronoUnit.class;
      boolean boolean0 = beanSerializerFactory0.isPotentialBeanType(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      Stack<BeanPropertyDefinition> stack0 = new Stack<BeanPropertyDefinition>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, annotationIntrospector0, false);
      stack0.add((BeanPropertyDefinition) pOJOPropertyBuilder0);
      beanSerializerFactory0.removeSetterlessGetters((SerializationConfig) null, (BeanDescription) null, stack0);
      assertTrue(stack0.isEmpty());
      assertEquals(0, stack0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      LinkedList<BeanPropertyWriter> linkedList0 = new LinkedList<BeanPropertyWriter>();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      linkedList0.add(beanPropertyWriter0);
      List<BeanPropertyWriter> list0 = beanSerializerFactory0.removeOverlappingTypeIds((SerializerProvider) null, (BeanDescription) null, (BeanSerializerBuilder) null, linkedList0);
      assertTrue(list0.contains(beanPropertyWriter0));
  }
}
