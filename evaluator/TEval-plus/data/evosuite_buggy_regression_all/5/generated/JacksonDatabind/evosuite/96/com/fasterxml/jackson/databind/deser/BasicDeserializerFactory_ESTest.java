/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:09:09 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BeanDeserializerModifier;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.deser.Deserializers;
import com.fasterxml.jackson.databind.deser.ValueInstantiators;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver;
import com.fasterxml.jackson.databind.node.TextNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.AccessPattern;
import java.sql.SQLRecoverableException;
import java.sql.SQLSyntaxErrorException;
import java.sql.SQLTransientConnectionException;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BasicDeserializerFactory_ESTest extends BasicDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      Deserializers.Base deserializers_Base0 = new Deserializers.Base();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAdditionalDeserializers(deserializers_Base0);
      assertFalse(deserializerFactory0.equals((Object)beanDeserializerFactory0));
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.withDeserializerModifier((BeanDeserializerModifier) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot pass null modifier
         //
         verifyException("com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<SQLSyntaxErrorException> class0 = SQLSyntaxErrorException.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JsonDeserializer<?> jsonDeserializer0 = beanDeserializerFactory0.findDefaultDeserializer(defaultDeserializationContext_Impl0, resolvedRecursiveType0, (BeanDescription) null);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      PlaceholderForType placeholderForType0 = new PlaceholderForType(1);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.resolveType(defaultDeserializationContext_Impl0, (BeanDescription) null, placeholderForType0, (AnnotatedMember) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = beanDeserializerFactory0.getFactoryConfig();
      assertFalse(deserializerFactoryConfig0.hasValueInstantiators());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      ValueInstantiators.Base valueInstantiators_Base0 = new ValueInstantiators.Base();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withValueInstantiators(valueInstantiators_Base0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._reportUnwrappedCreatorProperty((DeserializationContext) null, basicBeanDescription0, (AnnotatedParameter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      StdKeyDeserializers stdKeyDeserializers0 = new StdKeyDeserializers();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAdditionalKeyDeserializers(stdKeyDeserializers0);
      assertFalse(deserializerFactory0.equals((Object)beanDeserializerFactory0));
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAbstractTypeResolver(simpleAbstractTypeResolver0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ConcurrentHashMap> class0 = ConcurrentHashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      JavaType javaType0 = beanDeserializerFactory0.mapAbstractType((DeserializationConfig) null, mapType0);
      assertSame(javaType0, mapType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<SQLRecoverableException> class0 = SQLRecoverableException.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ArrayType arrayType0 = ArrayType.construct((JavaType) simpleType0, typeBindings0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createArrayDeserializer(defaultDeserializationContext_Impl0, arrayType0, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      Class<SQLTransientConnectionException> class1 = SQLTransientConnectionException.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createCollectionDeserializer(defaultDeserializationContext_Impl0, collectionType0, basicBeanDescription0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-4587));
      CollectionType collectionType0 = beanDeserializerFactory0._mapAbstractCollectionType(placeholderForType0, (DeserializationConfig) null);
      assertNull(collectionType0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashSet> class0 = HashSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createCollectionLikeDeserializer(defaultDeserializationContext_Impl0, collectionType0, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      Class<ConcurrentHashMap> class0 = ConcurrentHashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createMapDeserializer(defaultDeserializationContext_Impl0, mapType0, basicBeanDescription0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createKeyDeserializer(defaultDeserializationContext_Impl0, simpleType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      PlaceholderForType placeholderForType0 = new PlaceholderForType(3607);
      JsonDeserializer<Object> jsonDeserializer0 = beanDeserializerFactory0.createBeanDeserializer(defaultDeserializationContext_Impl0, placeholderForType0, (BeanDescription) null);
      assertEquals(AccessPattern.CONSTANT, jsonDeserializer0.getNullAccessPattern());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<SQLRecoverableException> class0 = SQLRecoverableException.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ArrayType arrayType0 = ArrayType.construct((JavaType) simpleType0, typeBindings0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createBeanDeserializer(defaultDeserializationContext_Impl0, arrayType0, (BeanDescription) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot deserialize Class [Ljava.sql.SQLRecoverableException; (of type array) as a Bean
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      JsonDeserializer<Object> jsonDeserializer0 = beanDeserializerFactory0.createBeanDeserializer(defaultDeserializationContext_Impl0, simpleType0, basicBeanDescription0);
      assertEquals(AccessPattern.CONSTANT, jsonDeserializer0.getEmptyAccessPattern());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<TextNode> class0 = TextNode.class;
      JsonDeserializer<?> jsonDeserializer0 = beanDeserializerFactory0._findCustomTreeNodeDeserializer(class0, (DeserializationConfig) null, (BeanDescription) null);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<SQLSyntaxErrorException> class0 = SQLSyntaxErrorException.class;
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      JsonDeserializer<?> jsonDeserializer0 = beanDeserializerFactory0._findCustomEnumDeserializer(class0, (DeserializationConfig) null, basicBeanDescription0);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<Boolean> class0 = Boolean.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.constructEnumResolver(class0, (DeserializationConfig) null, (AnnotatedMember) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ConcurrentHashMap> class0 = ConcurrentHashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._findJsonValueFor((DeserializationConfig) null, mapType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }
}
