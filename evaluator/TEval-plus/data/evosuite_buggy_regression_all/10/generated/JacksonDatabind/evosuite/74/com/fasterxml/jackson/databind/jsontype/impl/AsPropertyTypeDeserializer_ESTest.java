/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:38:04 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.InjectableValues;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import com.fasterxml.jackson.databind.util.TokenBuffer;
import java.io.IOException;
import java.time.chrono.ThaiBuddhistEra;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AsPropertyTypeDeserializer_ESTest extends AsPropertyTypeDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, " }0;DZa/>W+_hAZPT", true, (JavaType) null);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, (JavaType) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, (-1136), asPropertyTypeDeserializer0, propertyMetadata0);
      assertFalse(creatorProperty0.hasViews());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<ThaiBuddhistEra> class0 = ThaiBuddhistEra.class;
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class1, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(referenceType0, typeFactory0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.PROPERTY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(referenceType0, classNameIdResolver0, ",{Y+ZWL,r9h)a6Xl", false, referenceType0, jsonTypeInfo_As0);
      JsonTypeInfo.As jsonTypeInfo_As1 = asPropertyTypeDeserializer0.getTypeInclusion();
      assertSame(jsonTypeInfo_As0, jsonTypeInfo_As1);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "", true, simpleType0);
      TypeDeserializer typeDeserializer0 = asPropertyTypeDeserializer0.forProperty((BeanProperty) null);
      assertSame(typeDeserializer0, asPropertyTypeDeserializer0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "JSON", false, (JavaType) null);
      Object object0 = asPropertyTypeDeserializer0.deserializeTypedFromAny(jsonParser0, defaultDeserializationContext_Impl0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, (ConfigOverrides) null);
      InjectableValues.Std injectableValues_Std0 = new InjectableValues.Std();
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.createInstance(deserializationConfig0, jsonParser0, injectableValues_Std0);
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "JSON", false, (JavaType) null);
      try { 
        asPropertyTypeDeserializer0.deserializeTypedFromAny(jsonParser0, defaultDeserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected token (null), expected FIELD_NAME: missing property 'JSON' that is to contain type id  (for class com.fasterxml.jackson.databind.JsonDeserializer)
         //  at [Source: java.lang.String@0000000021; line: 1, column: 0]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer((JavaType) null, (TypeIdResolver) null, "JSON", true, (JavaType) null);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TokenBuffer tokenBuffer0 = new TokenBuffer(jsonParser0);
      Object object0 = asPropertyTypeDeserializer0._deserializeTypedUsingDefaultImpl(jsonParser0, defaultDeserializationContext_Impl0, tokenBuffer0);
      assertNull(object0);
  }
}