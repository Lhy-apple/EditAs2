/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:42:53 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.json.UTF8DataInputJsonParser;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BeanDeserializerModifier;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.deser.Deserializers;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver;
import com.fasterxml.jackson.databind.module.SimpleKeyDeserializers;
import com.fasterxml.jackson.databind.module.SimpleValueInstantiators;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import com.fasterxml.jackson.databind.util.TokenBuffer;
import java.sql.DataTruncation;
import java.sql.SQLClientInfoException;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.sql.SQLNonTransientConnectionException;
import java.sql.SQLTransientConnectionException;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BasicDeserializerFactory_ESTest extends BasicDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.withAdditionalDeserializers((Deserializers) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot pass null Deserializers
         //
         verifyException("com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      BeanDeserializerModifier beanDeserializerModifier0 = mock(BeanDeserializerModifier.class, new ViolatedAssumptionAnswer());
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withDeserializerModifier(beanDeserializerModifier0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<UTF8DataInputJsonParser> class0 = UTF8DataInputJsonParser.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.resolveType((DeserializationContext) null, basicBeanDescription0, (JavaType) null, (AnnotatedMember) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = beanDeserializerFactory0.getFactoryConfig();
      assertTrue(deserializerFactoryConfig0.hasKeyDeserializers());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      SimpleValueInstantiators simpleValueInstantiators0 = new SimpleValueInstantiators();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withValueInstantiators(simpleValueInstantiators0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._reportUnwrappedCreatorProperty(defaultDeserializationContext_Impl0, (BeanDescription) null, (AnnotatedParameter) null);
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
      SimpleKeyDeserializers simpleKeyDeserializers0 = new SimpleKeyDeserializers();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAdditionalKeyDeserializers(simpleKeyDeserializers0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAbstractTypeResolver(simpleAbstractTypeResolver0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ConcurrentSkipListMap> class0 = ConcurrentSkipListMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl1 = new DefaultDeserializationContext.Impl(defaultDeserializationContext_Impl0, deserializerFactory0);
      boolean boolean0 = defaultDeserializationContext_Impl1.hasValueDeserializerFor(mapType0, atomicReference0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<JsonLocation> class0 = JsonLocation.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeResolutionContext.Basic typeResolutionContext_Basic0 = new TypeResolutionContext.Basic(typeFactory0, typeBindings0);
      Class<SQLNonTransientConnectionException> class0 = SQLNonTransientConnectionException.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember(typeResolutionContext_Basic0, class0, "58cb-Dq##=", resolvedRecursiveType0);
      HashSet<SQLInvalidAuthorizationSpecException> hashSet0 = new HashSet<SQLInvalidAuthorizationSpecException>();
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._valueInstantiatorInstance((DeserializationConfig) null, virtualAnnotatedMember0, hashSet0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned key deserializer definition of type java.util.HashSet; expected type KeyDeserializer or Class<KeyDeserializer> instead
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      ValueInstantiator valueInstantiator0 = beanDeserializerFactory0._valueInstantiatorInstance((DeserializationConfig) null, (Annotated) null, (Object) null);
      assertNull(valueInstantiator0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, (DefaultDeserializationContext) null);
      PropertyAccessor propertyAccessor0 = PropertyAccessor.ALL;
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.NONE;
      objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      Class<SQLTransientConnectionException> class0 = SQLTransientConnectionException.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<DataTruncation> class0 = DataTruncation.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<SQLClientInfoException> class0 = SQLClientInfoException.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = TypeFactory.unknownType();
      JavaType[] javaTypeArray0 = new JavaType[0];
      MapType mapType0 = MapType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaType0, javaType0);
      CollectionType collectionType0 = beanDeserializerFactory0._mapAbstractCollectionType(mapType0, (DeserializationConfig) null);
      assertNull(collectionType0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayDeque> class0 = ArrayDeque.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
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
  public void test17()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayDeque> class0 = ArrayDeque.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(collectionType0, collectionType0, collectionType0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createMapLikeDeserializer(defaultDeserializationContext_Impl0, mapLikeType0, (BeanDescription) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<FloatNode> class0 = FloatNode.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.PROPERTY;
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.JAVA_LANG_OBJECT;
      objectMapper0.enableDefaultTyping(objectMapper_DefaultTyping0, jsonTypeInfo_As0);
      Class<Object> class0 = Object.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<TokenBuffer> class0 = TokenBuffer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ConcurrentSkipListMap> class0 = ConcurrentSkipListMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0, configOverrides0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._findJsonValueFor(deserializationConfig0, mapType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.cfg.MapperConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      AnnotatedMethod annotatedMethod0 = beanDeserializerFactory0._findJsonValueFor((DeserializationConfig) null, (JavaType) null);
      assertNull(annotatedMethod0);
  }
}
