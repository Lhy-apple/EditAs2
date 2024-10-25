/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:11:43 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.KeyDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
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
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver;
import com.fasterxml.jackson.databind.module.SimpleKeyDeserializers;
import com.fasterxml.jackson.databind.module.SimpleValueInstantiators;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.TokenBuffer;
import java.sql.BatchUpdateException;
import java.sql.SQLClientInfoException;
import java.sql.SQLDataException;
import java.sql.SQLNonTransientException;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BasicDeserializerFactory_ESTest extends BasicDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Deserializers.Base deserializers_Base0 = new Deserializers.Base();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAdditionalDeserializers(deserializers_Base0);
      assertNotSame(deserializerFactory0, beanDeserializerFactory0);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<ReaderBasedJsonParser> class0 = ReaderBasedJsonParser.class;
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
      assertFalse(deserializerFactoryConfig0.hasDeserializers());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      SimpleValueInstantiators simpleValueInstantiators0 = new SimpleValueInstantiators();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withValueInstantiators(simpleValueInstantiators0);
      assertNotSame(deserializerFactory0, beanDeserializerFactory0);
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
      Class<TreeMap> class0 = TreeMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(deserializerFactory0);
      boolean boolean0 = defaultDeserializationContext_Impl0.hasValueDeserializerFor(mapType0, atomicReference0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._valueInstantiatorInstance((DeserializationConfig) null, (Annotated) null, beanDeserializerFactory0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned key deserializer definition of type com.fasterxml.jackson.databind.deser.BeanDeserializerFactory; expected type KeyDeserializer or Class<KeyDeserializer> instead
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      ValueInstantiator valueInstantiator0 = beanDeserializerFactory0._valueInstantiatorInstance((DeserializationConfig) null, (Annotated) null, (Object) null);
      assertNull(valueInstantiator0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<AsWrapperTypeDeserializer> class0 = AsWrapperTypeDeserializer.class;
      JsonAutoDetect.Value jsonAutoDetect_Value0 = JsonAutoDetect.Value.defaultVisibility();
      ObjectMapper objectMapper1 = objectMapper0.setDefaultVisibility(jsonAutoDetect_Value0);
      ObjectReader objectReader0 = objectMapper1.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BatchUpdateException> class0 = BatchUpdateException.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
  public void test14()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<SQLNonTransientException> class0 = SQLNonTransientException.class;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createCollectionLikeDeserializer(defaultDeserializationContext_Impl0, collectionLikeType0, basicBeanDescription0);
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
      Class<ConcurrentSkipListMap> class0 = ConcurrentSkipListMap.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = TypeFactory.unknownType();
      JavaType[] javaTypeArray0 = new JavaType[0];
      MapType mapType0 = MapType.construct(class0, typeBindings0, javaType0, javaTypeArray0, javaType0, javaType0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createMapLikeDeserializer(defaultDeserializationContext_Impl0, mapType0, basicBeanDescription0);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<DoubleNode> class0 = DoubleNode.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.NON_CONCRETE_AND_ARRAYS;
      objectMapper0.enableDefaultTyping(objectMapper_DefaultTyping0);
      Class<SQLClientInfoException> class0 = SQLClientInfoException.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<TokenBuffer> class0 = TokenBuffer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<TreeMap> class0 = TreeMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapType0, typeFactory0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.WRAPPER_ARRAY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(mapType0, classNameIdResolver0, "wDt/*MN.%'E", true, mapType0, jsonTypeInfo_As0);
      JsonDeserializer<SQLDataException> jsonDeserializer0 = (JsonDeserializer<SQLDataException>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<?> jsonDeserializer1 = beanDeserializerFactory0._findCustomMapLikeDeserializer(mapType0, (DeserializationConfig) null, basicBeanDescription0, (KeyDeserializer) null, asPropertyTypeDeserializer0, jsonDeserializer0);
      assertNull(jsonDeserializer1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      JavaType javaType0 = TypeFactory.unknownType();
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._findJsonValueFor((DeserializationConfig) null, javaType0);
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
      AnnotatedMethod annotatedMethod0 = beanDeserializerFactory0._findJsonValueFor((DeserializationConfig) null, (JavaType) null);
      assertNull(annotatedMethod0);
  }
}
