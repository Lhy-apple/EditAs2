/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:05:27 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.UnresolvedForwardReference;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.impl.ReadableObjectId;
import com.fasterxml.jackson.databind.deser.std.CollectionDeserializer;
import com.fasterxml.jackson.databind.deser.std.JsonLocationInstantiator;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.io.InputStream;
import java.time.chrono.ChronoLocalDate;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeSet;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CollectionDeserializer_ESTest extends CollectionDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      CollectionDeserializer.CollectionReferringAccumulator collectionDeserializer_CollectionReferringAccumulator0 = new CollectionDeserializer.CollectionReferringAccumulator(class0, linkedHashSet0);
      JsonLocation jsonLocation0 = JsonLocation.NA;
      ObjectIdGenerator.IdKey objectIdGenerator_IdKey0 = new ObjectIdGenerator.IdKey(class0, class0, linkedHashSet0);
      ReadableObjectId readableObjectId0 = new ReadableObjectId(objectIdGenerator_IdKey0);
      UnresolvedForwardReference unresolvedForwardReference0 = new UnresolvedForwardReference((JsonParser) null, "ts=F:(AmGCWF", jsonLocation0, readableObjectId0);
      ReadableObjectId.Referring readableObjectId_Referring0 = collectionDeserializer_CollectionReferringAccumulator0.handleUnresolvedReference(unresolvedForwardReference0);
      collectionDeserializer_CollectionReferringAccumulator0.resolveForwardReference(linkedHashSet0, readableObjectId_Referring0);
      assertEquals(1, linkedHashSet0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0);
      // Undeclared exception!
      try { 
        collectionDeserializer0.findBackReference(": value instantiator (");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not handle managed/back reference ': value instantiator (': type: container deserializer of type com.fasterxml.jackson.databind.deser.std.CollectionDeserializer returned null for 'getContentDeserializer()'
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.ContainerDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      // Undeclared exception!
      try { 
        collectionDeserializer0.deserializeWithType(jsonParser0, deserializationContext0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.CollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CollectionDeserializer collectionDeserializer0 = null;
      try {
        collectionDeserializer0 = new CollectionDeserializer((CollectionDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.CollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      JavaType[] javaTypeArray0 = new JavaType[0];
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) referenceType0, javaTypeArray0, (JavaType) referenceType0);
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, 62);
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Boolean boolean0 = Boolean.TRUE;
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer(collectionType0, coreXMLDeserializers_Std0, (TypeDeserializer) null, jsonLocationInstantiator0, coreXMLDeserializers_Std0, boolean0);
      JavaType javaType0 = collectionDeserializer0.getContentType();
      assertSame(javaType0, referenceType0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Map> class0 = Map.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      Class<Integer> class1 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ArrayType arrayType0 = ArrayType.construct((JavaType) mapType0, typeBindings0, (Object) class1, (Object) class1);
      Class<Object> class2 = Object.class;
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(arrayType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(mapType0, classNameIdResolver0, (String) null, false, class2);
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer(arrayType0, (JsonDeserializer<Object>) null, asExternalTypeDeserializer0, jsonLocationInstantiator0);
      collectionDeserializer0.withResolved(collectionDeserializer0, collectionDeserializer0, asExternalTypeDeserializer0);
      assertFalse(collectionDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<MapType> class0 = MapType.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0);
      Boolean boolean0 = Boolean.valueOf(true);
      CollectionDeserializer collectionDeserializer1 = collectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (TypeDeserializer) null, boolean0);
      assertNotSame(collectionDeserializer1, collectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class1, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(resolvedRecursiveType0, typeFactory0);
      Class<Integer> class2 = Integer.class;
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(resolvedRecursiveType0, classNameIdResolver0, "cZ3:1b", true, class2);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      MapType mapType0 = MapType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) collectionLikeType0, (JavaType) resolvedRecursiveType0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, resolvedRecursiveType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 1840, mapType0, propertyMetadata0);
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer(resolvedRecursiveType0, (JsonDeserializer<Object>) null, asArrayTypeDeserializer0, (ValueInstantiator) null);
      TypeDeserializer typeDeserializer0 = asArrayTypeDeserializer0.forProperty(creatorProperty0);
      CollectionDeserializer collectionDeserializer1 = collectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, typeDeserializer0, (Boolean) null);
      assertFalse(collectionDeserializer1.isCachable());
      assertNotSame(collectionDeserializer1, collectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0);
      CollectionDeserializer collectionDeserializer1 = collectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (TypeDeserializer) null);
      assertSame(collectionDeserializer1, collectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      JavaType[] javaTypeArray0 = new JavaType[0];
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) referenceType0, javaTypeArray0, (JavaType) referenceType0);
      CoreXMLDeserializers.Std coreXMLDeserializers_Std0 = new CoreXMLDeserializers.Std(class0, 62);
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      Boolean boolean0 = Boolean.TRUE;
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer(collectionType0, coreXMLDeserializers_Std0, (TypeDeserializer) null, jsonLocationInstantiator0, coreXMLDeserializers_Std0, boolean0);
      boolean boolean1 = collectionDeserializer0.isCachable();
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Map> class0 = Map.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      Class<Integer> class1 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ArrayType arrayType0 = ArrayType.construct((JavaType) mapType0, typeBindings0, (Object) class1, (Object) class1);
      Class<Object> class2 = Object.class;
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(arrayType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(mapType0, classNameIdResolver0, (String) null, false, class2);
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer(arrayType0, (JsonDeserializer<Object>) null, asExternalTypeDeserializer0, jsonLocationInstantiator0);
      boolean boolean0 = collectionDeserializer0.isCachable();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class1, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(resolvedRecursiveType0, typeFactory0);
      Class<Integer> class2 = Integer.class;
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(resolvedRecursiveType0, classNameIdResolver0, ") out of range of long (", true, class2);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      MapType mapType0 = MapType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) collectionLikeType0, (JavaType) resolvedRecursiveType0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, resolvedRecursiveType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 1840, mapType0, propertyMetadata0);
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer(resolvedRecursiveType0, (JsonDeserializer<Object>) null, asArrayTypeDeserializer0, (ValueInstantiator) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        collectionDeserializer0.createContextual(defaultDeserializationContext_Impl0, creatorProperty0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<MapType> class0 = MapType.class;
      ObjectMapper objectMapper1 = objectMapper0.enableDefaultTyping();
      ObjectReader objectReader0 = objectMapper1.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      try { 
        collectionDeserializer0.deserialize(jsonParser0, deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not instantiate value of type com.fasterxml.jackson.core.JsonLocation; no default creator found
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      TreeSet<Object> treeSet0 = new TreeSet<Object>();
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, false, false);
      filteringParserDelegate0.nextBooleanValue();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0, (JsonDeserializer<Object>) null, (Boolean) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        collectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<Object>) treeSet0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.CollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      TreeSet<Object> treeSet0 = new TreeSet<Object>();
      Boolean boolean0 = Boolean.TRUE;
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0, (JsonDeserializer<Object>) null, boolean0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      try { 
        collectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<Object>) treeSet0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // (was java.lang.NullPointerException) (through reference chain: java.lang.Object[0])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      TreeSet<Object> treeSet0 = new TreeSet<Object>();
      Boolean boolean0 = Boolean.FALSE;
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0, (JsonDeserializer<Object>) null, boolean0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        collectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<Object>) treeSet0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.CollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonLocationInstantiator jsonLocationInstantiator0 = new JsonLocationInstantiator();
      CollectionDeserializer collectionDeserializer0 = new CollectionDeserializer((JavaType) null, (JsonDeserializer<Object>) null, (TypeDeserializer) null, jsonLocationInstantiator0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      Vector<Object> vector0 = new Vector<Object>();
      // Undeclared exception!
      try { 
        collectionDeserializer0.deserialize(jsonParser0, deserializationContext0, (Collection<Object>) vector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.CollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<InputStream> class0 = InputStream.class;
      PriorityQueue<Object> priorityQueue0 = new PriorityQueue<Object>();
      CollectionDeserializer.CollectionReferringAccumulator collectionDeserializer_CollectionReferringAccumulator0 = new CollectionDeserializer.CollectionReferringAccumulator(class0, priorityQueue0);
      collectionDeserializer_CollectionReferringAccumulator0.handleUnresolvedReference((UnresolvedForwardReference) null);
      collectionDeserializer_CollectionReferringAccumulator0.add(objectMapper0);
      assertEquals(0, priorityQueue0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      CollectionDeserializer.CollectionReferringAccumulator collectionDeserializer_CollectionReferringAccumulator0 = new CollectionDeserializer.CollectionReferringAccumulator(class0, (Collection<Object>) null);
      // Undeclared exception!
      try { 
        collectionDeserializer_CollectionReferringAccumulator0.add(class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.CollectionDeserializer$CollectionReferringAccumulator", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      CollectionDeserializer.CollectionReferringAccumulator collectionDeserializer_CollectionReferringAccumulator0 = new CollectionDeserializer.CollectionReferringAccumulator(class0, linkedHashSet0);
      JsonLocation jsonLocation0 = JsonLocation.NA;
      ObjectIdGenerator.IdKey objectIdGenerator_IdKey0 = new ObjectIdGenerator.IdKey(class0, class0, linkedHashSet0);
      ReadableObjectId readableObjectId0 = new ReadableObjectId(objectIdGenerator_IdKey0);
      UnresolvedForwardReference unresolvedForwardReference0 = new UnresolvedForwardReference((JsonParser) null, "ts=F:(AmGCWF", jsonLocation0, readableObjectId0);
      collectionDeserializer_CollectionReferringAccumulator0.handleUnresolvedReference(unresolvedForwardReference0);
      // Undeclared exception!
      try { 
        collectionDeserializer_CollectionReferringAccumulator0.resolveForwardReference(class0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Trying to resolve a forward reference with id [class java.io.InputStream] that wasn't previously seen as unresolved.
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.CollectionDeserializer$CollectionReferringAccumulator", e);
      }
  }
}